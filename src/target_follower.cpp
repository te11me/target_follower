
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/objdetect.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Bool.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_listener.h>
#include <dynamic_reconfigure/server.h>
#include <target_follower/TargetFollowerConfig.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <mutex>
#include <thread>
#include <fstream>
#include <boost/filesystem.hpp>
#include <deque>
#include <memory>
#include <NvInferRuntime.h> // 添加头文件
#include <nav_msgs/GetMap.h>
#include <numeric>                       // 必须包含此头文件
// 强制使用自定义OpenCV路径
#define OPENCV_TRAITS_ENABLE_DEPRECATED
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/tracking.hpp> // OpenCV4.5.5跟踪器头文件


using namespace cv;
using namespace std;
using namespace nvinfer1;
namespace fs = boost::filesystem;

// TensorRT Logger
class Logger : public ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            ROS_WARN("TensorRT: %s", msg);
    }
} gLogger;

class TRTFeatureExtractor
{
private:
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    cudaStream_t stream;
    void *buffers[2];
    int inputIndex, outputIndex;
    size_t inputSize, outputSize;

    struct Destroy
    {
        void operator()(nvinfer1::ICudaEngine *ptr) { ptr->destroy(); }
        void operator()(nvinfer1::IExecutionContext *ptr) { ptr->destroy(); }
    };
    std::unique_ptr<nvinfer1::ICudaEngine, Destroy> engine_holder;
    std::unique_ptr<nvinfer1::IExecutionContext, Destroy> context_holder;

public:
    TRTFeatureExtractor(const string &engine_path)
    {
        std::ifstream engineFile(engine_path, std::ios::binary);
        engineFile.seekg(0, ios::end);
        size_t size = engineFile.tellg();
        engineFile.seekg(0, ios::beg);
        vector<char> engineData(size);
        engineFile.read(engineData.data(), size);

        IRuntime *runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
        context = engine->createExecutionContext();

        inputIndex = engine->getBindingIndex("input");
        outputIndex = engine->getBindingIndex("output");

        Dims inputDims = engine->getBindingDimensions(inputIndex);
        inputSize = 1;
        for (int i = 0; i < inputDims.nbDims; i++)
            inputSize *= inputDims.d[i];

        Dims outputDims = engine->getBindingDimensions(outputIndex);
        outputSize = 1;
        for (int i = 0; i < outputDims.nbDims; i++)
            outputSize *= outputDims.d[i];

        cudaMalloc(&buffers[inputIndex], inputSize * sizeof(float));
        cudaMalloc(&buffers[outputIndex], outputSize * sizeof(float));
        cudaStreamCreate(&stream);

        engine_holder.reset(engine);
        context_holder.reset(context);
    }

    vector<float> extractFeature(const Mat &img)
    {
        Mat resized, floatImg;
        resize(img, resized, Size(224, 224));
        resized.convertTo(floatImg, CV_32FC3, 1 / 255.0);

        vector<Mat> chw;
        split(floatImg, chw);
        vector<float> data(inputSize);
        for (int c = 0; c < 3; c++)
        {
            memcpy(data.data() + c * 224 * 224, chw[c].data, 224 * 224 * sizeof(float));
        }

        cudaMemcpyAsync(buffers[inputIndex], data.data(),
                        inputSize * sizeof(float), cudaMemcpyHostToDevice, stream);
        context->enqueueV2(buffers, stream, nullptr);

        vector<float> output(outputSize);
        cudaMemcpyAsync(output.data(), buffers[outputIndex],
                        outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        return output;
    }

    ~TRTFeatureExtractor()
    {
        cudaFree(buffers[inputIndex]);
        cudaFree(buffers[outputIndex]);
        cudaStreamDestroy(stream);
    }
};
class TargetFollower
{
private:
    ros::NodeHandle nh;
    image_transport::ImageTransport it;
    ros::Timer save_timer; // 定时器
    bool tracking = false; // 跟踪状态标志
    cv::Mat current_frame;

    // Subscribers
    image_transport::Subscriber proc_img_sub;
    ros::Subscriber lidar_sub, start_sub, stop_sub, emergency_sub;

    // Publishers
    ros::Publisher cmd_vel_pub, finish_pub;
    image_transport::Publisher debug_img_pub;
    ros::Publisher debug_info_pub;

    // Tracking
    struct TrackingState
    {
        Rect2d roi;
        vector<float> features;
        bool active = false;
        int lost_frames = 0;
        Mat last_roi_img;
        deque<float> confidence_history;
    } track_state;

    // cv::Ptr<cv::Tracker> tracker;
    // OpenCV3.4.5的跟踪器初始化方式：
    // tracker = cv::TrackerCSRT::create(); // 使用具体实现类的create方法
    cv::Ptr<cv::TrackerCSRT> tracker; // 使用具体跟踪器类型
    // cv::Ptr<cv::Tracker> tracker;
    mutex track_mutex;

    // Configuration
    double max_linear = 0.5, max_angular = 1.0;
    double follow_dist = 0.6, obstacle_dist = 0.25;
    bool avoid_obstacle = true, emergency_stop = false;

    // CUDA加速
    cv::cuda::GpuMat gpu_frame, gpu_hsv, gpu_mask;

    // TensorRT
    unique_ptr<TRTFeatureExtractor> feature_extractor;

    // Debug
    struct DebugConfig
    {
        bool show_images = false;
        bool save_images = false;
        int log_level = 1;
    } debug_config;

    // System
    tf::TransformListener tf_listener;
    thread control_thread, safety_thread;
    string save_path = "/tmp/target_rois";
    double save_interval = 0.25;
    double save_duration = 10.0;
    atomic<bool> saving_images{false};
    int total_saves = 0;

public:
    TargetFollower() : it(nh)
    {
        nh.param("save_path", save_path, save_path);
        nh.param("save_interval", save_interval, save_interval);
        nh.param("save_duration", save_duration, save_duration);
        nh.param("debug/show_images", debug_config.show_images, false);
        nh.param("debug/log_level", debug_config.log_level, 1);

        // 初始化跟踪器
        // tracker = cv::Tracker::create("CSRT");
        // tracker = cv::Tracker::update();
        // 初始化跟踪器（OpenCV4.5.5正确初始���方式）
        tracker = cv::TrackerCSRT::create(); // 使用CSRT跟踪器

        // 订阅话题

        // 订阅话题
        proc_img_sub = it.subscribe("/processed_image", 1,
                                    &TargetFollower::imageCallback, this);

        lidar_sub = nh.subscribe("/scan", 1,
                                 &TargetFollower::lidarCallback, this);
        start_sub = nh.subscribe("/start_detect", 1,
                                 &TargetFollower::startCallback, this);
        stop_sub = nh.subscribe("/stop_follow", 1,
                                &TargetFollower::stopCallback, this);
        emergency_sub = nh.subscribe("/emergency_stop", 1,
                                     &TargetFollower::emergencyCallback, this);

        // 发布话题
        cmd_vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
        finish_pub = nh.advertise<std_msgs::Bool>("/finish_detect", 1);
        debug_img_pub = it.advertise("/debug/tracking_view", 1);
        debug_info_pub = nh.advertise<geometry_msgs::Twist>("/debug/control_info", 10);

        // 加载TensorRT引擎
        string engine_path;
        nh.param<string>("trt_engine", engine_path, "mobilenet_v2.engine");
        feature_extractor = make_unique<TRTFeatureExtractor>(engine_path);

        // 启动控制线程
        control_thread = thread(&TargetFollower::controlLoop, this);
        safety_thread = thread(&TargetFollower::safetyMonitor, this);

        ROS_INFO_COND(debug_config.log_level >= 1,
                      "Target Follower Node initialized with debug level %d",
                      debug_config.log_level);
    }

    void lidarCallback(const sensor_msgs::LaserScanConstPtr &msg) {}
    // void startCallback(const std_msgs::Bool::ConstPtr& msg) {}
    // void stopCallback(const std_msgs::Bool::ConstPtr& msg) {}
    void emergencyCallback(const std_msgs::Bool::ConstPtr &msg)
    {
        emergency_stop = msg->data;
        if (emergency_stop)
        {
            geometry_msgs::Twist cmd;
            cmd_vel_pub.publish(cmd);
        }
    }

    void imageCallback(const sensor_msgs::ImageConstPtr &msg)
    {
        if (emergency_stop)
            return;

        try
        {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            current_frame = cv_ptr->image.clone();
            Mat frame = cv_ptr->image;

            // CUDA加速处理
            gpu_frame.upload(frame);
            cv::cuda::cvtColor(gpu_frame, gpu_hsv, COLOR_BGR2HSV);
            // cv::cuda::inRange(gpu_hsv, Scalar(35,50,50), Scalar(85,255,255), gpu_mask);
            // 修改为CPU版本（因OpenCV3.4.5的cuda模块无inRange）：
            cv::Mat hsv, cpu_mask;
            cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
            cv::inRange(hsv, cv::Scalar(35, 50, 50), cv::Scalar(85, 255, 255), cpu_mask);
            // gpu_mask.upload(cpu_mask); // 若仍需GPU处理
            // Mat cpu_mask;
            // cv::cuda::GpuMat gpu_mask;
            // cv::cuda::inRange(gpu_hsv, Scalar(35, 50, 50), Scalar(85, 255, 255), gpu_mask);
            // gpu_mask.download(cpu_mask);

            // 更新跟踪状态
            updateTracking(frame, cpu_mask);

            // 保存调试图像
            publishDebugImage(cv_ptr);

            if (debug_config.log_level >= 2)
            {
                double min_val, max_val;
                minMaxLoc(cpu_mask, &min_val, &max_val);
                ROS_DEBUG_THROTTLE(1.0,
                                   "[Depth] Min: %.2fm, Max: %.2fm, Valid: %d/%d",
                                   min_val, max_val,
                                   //    countNonZero(cpu_mask),
                                   //    cpu_mask.total());
                                   (int)countNonZero(cpu_mask),
                                   (int)cpu_mask.total());
            }
        }
        catch (cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

    void updateTracking(const Mat &frame, const Mat &mask)
    {
        lock_guard<mutex> lock(track_mutex);

        if (track_state.active)
        {
            // 修改为正确的update调用方式
            cv::Rect updated_roi; // 将类型改为cv::Rect
            bool success = tracker->update(frame, updated_roi);  // 正确方法
            
            track_state.roi = cv::Rect2d(updated_roi);  // 将cv::Rect转换为cv::Rect2d存储
            track_state.confidence_history.push_back(success ? 1.0f : 0.0f);

            if (track_state.confidence_history.size() > 5)
                track_state.confidence_history.pop_front();

            float avg_confidence = std::accumulate(
                                       track_state.confidence_history.begin(),
                                       track_state.confidence_history.end(), 0.0f) /
                                   track_state.confidence_history.size();

            if (avg_confidence < 0.4)
            {
                if (++track_state.lost_frames > 5)
                {
                    track_state.active = false;
                    ROS_WARN_COND(debug_config.log_level >= 1,
                                  "Target lost! Last confidence: %.2f", avg_confidence);
                }
                return;
            }

            Mat current_roi = frame(track_state.roi);
            auto current_features = feature_extractor->extractFeature(current_roi);
            float similarity = cosineSimilarity(
                track_state.features, current_features);

            if (similarity < 0.7)
            {
                ROS_WARN_COND(debug_config.log_level >= 1,
                              "Target mismatch detected! Similarity: %.2f", similarity);
                track_state.active = false;
                return;
            }

            track_state.lost_frames = 0;

            if (debug_config.log_level >= 2)
            {
                ROS_DEBUG("[Tracking] ROI: (%.0f,%.0f) %fx%f, Conf: %.2f, Sim: %.2f",
                          track_state.roi.x, track_state.roi.y,
                          track_state.roi.width, track_state.roi.height,
                          avg_confidence, similarity);
            }
        }
        else if (!track_state.last_roi_img.empty()) 
        {
            vector<Rect> found;
            cv::CascadeClassifier detector;
            detector.detectMultiScale(frame, found, 1.1, 3, 0, Size(30, 30));

            for (const auto &r : found)
            {
                Mat candidate = frame(r);
                auto features = feature_extractor->extractFeature(candidate);
                float similarity = cosineSimilarity(
                    track_state.features, features);
                // 修正点：移除重复的初始化调用，统一使用Rect2d
                tracker = cv::TrackerCSRT::create();  // 关键点：必须重新创建跟踪器
                tracker->init(frame, cv::Rect2d(r));  // 确保使用Rect2d初始化
                track_state.active = true;
                track_state.roi = cv::Rect2d(r);
                ROS_INFO_COND(debug_config.log_level >= 1,
                            "Target reacquired! Similarity: %.2f", similarity);
                break;
            }
        }
    }

    void controlLoop()
    {
        ros::Rate rate(30);
        while (ros::ok())
        {
            if (emergency_stop)
            {
                rate.sleep();
                continue;
            }

            geometry_msgs::Twist cmd;
            double current_distance = 0.0;
            double current_angle = 0.0;

            try
            {
                tf::StampedTransform transform;
                tf_listener.lookupTransform("base_link", "target",
                                            ros::Time(0), transform);

                current_angle = atan2(transform.getOrigin().y(),
                                      transform.getOrigin().x());
                current_distance = transform.getOrigin().x();

                double safe_distance = follow_dist - 0.1;
                if (current_distance < safe_distance && current_distance > 0.3)
                {
                    cmd.linear.x = 0;
                }
                else if (current_distance <= 0.3)
                {
                    cmd.linear.x = -clamp(0.3 - current_distance, 0.0, 0.1);
                    if (checkRearObstacles())
                    {
                        cmd.linear.x = 0;
                        ROS_WARN_COND(debug_config.log_level >= 1,
                                      "Rear obstacle detected! Stop backing up");
                    }
                }
                else
                {
                    cmd.linear.x = clamp(current_distance - follow_dist,
                                         -0.1, max_linear);
                }

                cmd.angular.z = clamp(current_angle * 0.5,
                                      -max_angular, max_angular);
            }
            catch (tf::TransformException &ex)
            {
                ROS_DEBUG_THROTTLE(1.0, "TF Error: %s", ex.what());
            }

            if (avoid_obstacle && checkFrontObstacles())
            {
                cmd.linear.x = 0;
                ROS_WARN_COND(debug_config.log_level >= 1,
                              "Front obstacle detected!");
            }

            cmd_vel_pub.publish(cmd);
            publishDebugInfo(cmd, current_distance, current_angle);
            rate.sleep();
        }
    }

    bool checkFrontObstacles()
    {
        auto scan = ros::topic::waitForMessage<sensor_msgs::LaserScan>("/scan", nh);
        int center = scan->ranges.size() / 2;
        int range = scan->ranges.size() * 60 / 360 / 2;
        for (int i = center - range; i <= center + range; ++i)
        {
            if (scan->ranges[i] < obstacle_dist)
                return true;
        }
        return false;
    }

    bool checkRearObstacles()
    {
        auto scan = ros::topic::waitForMessage<sensor_msgs::LaserScan>("/scan", nh);
        int rear_start = scan->ranges.size() * 5 / 8;
        int rear_end = scan->ranges.size() * 7 / 8;
        for (int i = rear_start; i < rear_end; ++i)
        {
            if (scan->ranges[i] < obstacle_dist + 0.1)
                return true;
        }
        return false;
    }

    void safetyMonitor()
    {
        ros::Rate rate(50);
        while (ros::ok())
        {
            if (readCpuTemp() > 85.0)
            {
                ROS_FATAL("System overheating! Shutting down...");
                exit(EXIT_FAILURE);
            }
            rate.sleep();
        }
    }

private:
    void publishDebugImage(const cv_bridge::CvImagePtr &cv_ptr)
    {
        if (debug_config.show_images || debug_config.save_images)
        {
            Mat debug_img;
            cv::cvtColor(cv_ptr->image, debug_img, COLOR_BGR2RGB);

            if (track_state.active)
            {
                rectangle(debug_img, track_state.roi, Scalar(255, 0, 0), 2);
                putText(debug_img,
                        format("Conf: %.2f", track_state.confidence_history.back()),
                        Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8,
                        Scalar(0, 255, 0), 2);
            }

            if (debug_config.show_images)
            {
                cv_bridge::CvImage debug_msg;
                debug_msg.header = cv_ptr->header;
                debug_msg.encoding = "rgb8";
                debug_msg.image = debug_img;
                debug_img_pub.publish(debug_msg.toImageMsg());
            }

            if (debug_config.save_images)
            {
                static int save_count = 0;
                string filename = save_path + "/debug_" + to_string(save_count++) + ".jpg";
                imwrite(filename, debug_img);
            }
        }
    }

    void publishDebugInfo(const geometry_msgs::Twist &cmd,
                          double distance, double angle)
    {
        debug_info_pub.publish(cmd);

        if (debug_config.log_level >= 1)
        {
            ROS_INFO_THROTTLE(0.5,
                              "[Control] Speed: %.2fm/s, Steering: %.2frad, Dist: %.2fm",
                              cmd.linear.x, cmd.angular.z, distance);
        }
    }

    float readCpuTemp()
    {
        ifstream temp_file("/sys/class/thermal/thermal_zone0/temp");
        float temp;
        temp_file >> temp;
        return temp / 1000.0;
    }

    float cosineSimilarity(const vector<float> &v1, const vector<float> &v2)
    {
        float dot = 0, norm1 = 0, norm2 = 0;
        for (size_t i = 0; i < v1.size(); ++i)
        {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }
        return dot / (sqrt(norm1) * sqrt(norm2));
        // 接续cosineSimilarity函数闭合
    }

    template <typename T>
    T clamp(const T &val, const T &min, const T &max)
    {
        return std::max(min, std::min(val, max));
    }

    // 补全缺失的回调函数实现
    void startCallback(const std_msgs::Bool::ConstPtr &msg)
    {
        if (msg->data && !emergency_stop)
        {
            fs::remove_all(save_path);
            fs::create_directories(save_path);

            saving_images = true;
            total_saves = 0;
            save_timer = nh.createTimer(ros::Duration(save_interval),
                                        &TargetFollower::saveImageHandler, this);
            ros::Timer finish_timer = nh.createTimer(ros::Duration(save_duration), [this](const ros::TimerEvent &)
                                                     { 
                    saving_images = false;
                    finish_pub.publish(std_msgs::Bool()); }, true);
        }
    }

    void stopCallback(const std_msgs::Bool::ConstPtr &msg)
    {
        if (msg->data)
        {
            tracking = false;
            saveMap();
        }
    }

    void saveImageHandler(const ros::TimerEvent &)
    {
        if (!saving_images || total_saves >= (save_duration / save_interval))
            return;

        lock_guard<mutex> lock(track_mutex);
        if (track_state.roi.empty())
            return;

        // Mat roi_img = current_frame(track_state.roi).clone();
        // 修正为：
        Mat roi_img = current_frame(track_state.roi).clone(); // frame是当前处理的图像
        string filename = save_path + "/roi_" + to_string(total_saves++) + ".png";
        imwrite(filename, roi_img);
        track_state.last_roi_img = roi_img.clone();
    }

    void saveMap()
    {
        nav_msgs::GetMap srv;
        if (ros::service::call("/dynamic_map", srv))
        {
            string map_path = save_path + "/saved_map";
            cv::Mat map_img(srv.response.map.info.height,
                            srv.response.map.info.width,
                            CV_8UC1, &srv.response.map.data[0]);
            cv::imwrite(map_path + ".pgm", map_img);

            ofstream yaml(map_path + ".yaml");
            yaml << "image: " << map_path << ".pgm\n"
                 << "resolution: " << srv.response.map.info.resolution << "\n"
                 << "origin: [" << srv.response.map.info.origin.position.x << ","
                 << srv.response.map.info.origin.position.y << ",0]\n";
            yaml.close();
        }
    }
}; // TargetFollower类闭合

// 主函数保持不变
int main(int argc, char **argv)
{
    ros::init(argc, argv, "target_follower");

    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    if (sched_setscheduler(0, SCHED_FIFO, &param) == -1)
    {
        ROS_WARN("Failed to set real-time scheduler");
    }

    TargetFollower follower;
    ros::spin();
    return 0;
}