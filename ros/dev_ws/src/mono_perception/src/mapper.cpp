#include "ros/ros.h"
#include "nav_msgs/OccupancyGrid.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/TransformStamped.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <ros/exception.h>

/*
TODO: 
- Get parameters from parameter server such as:
    * linear/angular update
    * map size
    * 

- get resolution and others from local grid messages
*/

class SemanticMapper
{
public:
    SemanticMapper() : nh("~")
    {
        odom_sub = nh.subscribe("/odom", 10, &SemanticMapper::odomCallback, this);
        bev_rgb_sub = nh.subscribe("/local_occ_grid", 10, &SemanticMapper::bev_cb, this);
        semantic_map_pub = nh.advertise<nav_msgs::OccupancyGrid>("/global_occ_grid", 10);

        // Initialize global occupancy grid parameters
        resolution = 1.0 / 80.0; // (meters per pixel)
        map_size = static_cast<int>(80.0 / resolution); // map size (in pixels)
        global_occ_grid = std::vector<int8_t>(map_size * map_size, -1);

        linear_update = 0.05;
        angular_update = 0.02;

        // Initialize robot pose
        robot_pose = {0.0, 0.0, 0.0}; // TODO: change initialization
        last_robot_pose = robot_pose;

        // Set up TF broadcaster
        tf_broadcaster = new tf2_ros::TransformBroadcaster();

        ROS_INFO_STREAM("Started mapping node with map size of " << map_size << "x" << map_size << " [m2]");
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr &odom_msg)
    {
        // Update robot pose based on odometry
        tf2::Quaternion quat;
        tf2::fromMsg(odom_msg->pose.pose.orientation, quat);
        double roll, pitch, yaw;
        tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);

        robot_pose[0] = odom_msg->pose.pose.position.x;
        robot_pose[1] = odom_msg->pose.pose.position.y;
        robot_pose[2] = yaw;
    }

    void bev_cb(const nav_msgs::OccupancyGrid::ConstPtr &local_occ_grid_msg)
    {
        // Transform local occupancy grid to global coordinates
        if (checkForUpdate())
        {
            auto local_occ_grid = local_occ_grid_msg->data;
            auto global_position = worldToMapIndices(0);

            updateMap(global_occ_grid, local_occ_grid, global_position, robot_pose[2]);
    
            // Publish the updated global occupancy grid
            last_robot_pose = robot_pose;

            ROS_INFO_STREAM("Updated map at position x=" << robot_pose[0] << ", y=" << robot_pose[1]);
        }

        publishGlobalOccGrid();
        publishTransform();
    }

    std::vector<int8_t> rot90(const std::vector<int8_t>& input, int x_dim, int y_dim)
    {
        std::vector<int8_t> rotated_input;
        rotated_input.reserve(input.size());

        for (int x_ = x_dim - 1; x_ >= 0; --x_)
        {
            for (int y_ = 0; y_ < y_dim; ++y_)
            {
                rotated_input.push_back(input[y_ * y_dim + x_]);
            }
        }

        return rotated_input;
    }

    void updateMap(std::vector<int8_t> &map, const std::vector<int8_t> &sensor_data, const std::vector<int> &position, double yaw)
    {
        int Ll = static_cast<int>(sqrt(sensor_data.size()));
        int Hl = Ll;
        int xR = position[0];
        int yR = position[1];
        double yawR = yaw;

        int dx = static_cast<int>(3.25 * 80);
        int dy = Ll / 2;

        yawR *= -1;

        auto rotated_sensor_data = rot90(sensor_data, Ll, Hl);

        for (int x_ = 0; x_ < Ll; ++x_)
        {
            for (int y_ = 0; y_ < Hl; ++y_)
            {
                if (rotated_sensor_data[y_ * Hl + x_] == -1)
                    continue;

                double R = sqrt(pow(Hl - y_ + dy, 2) + pow(x_ - dx, 2));
                double phi = atan2(x_ - dx, Hl - y_ + dy);

                int rx = static_cast<int>(R * cos(yawR - phi)) - map_size / 2;
                int ry = static_cast<int>(R * sin(yawR - phi)) - map_size / 2;

                int i = xR + rx;
                int j = yR - ry;

                if (rotated_sensor_data[y_ * Hl + x_] == 0)
                {
                    map[j * map_size + i] = 0;
                }
                else
                {
                    map[j * map_size + i] += static_cast<int>(rotated_sensor_data[y_ * Hl + x_]);
                }

                // clamp values to range 0, 100 (at this point we ignored unkown space "-1")
                map[j * map_size + i] = std::max(std::min(std::abs(map[j * map_size + i]), 100), 0);
            }
        }
    }

    std::vector<int> worldToMapIndices(int origin)
    {
        // Convert world coordinates to map indices
        int x_idx = static_cast<int>((robot_pose[0] - 20) / resolution);
        int y_idx = static_cast<int>((robot_pose[1] - 10) / resolution);

        return {x_idx, y_idx};
    }

    void publishGlobalOccGrid()
    {
        // Publish the updated global occupancy grid
        nav_msgs::OccupancyGrid global_occ_grid_msg;
        global_occ_grid_msg.header.stamp = ros::Time::now();
        global_occ_grid_msg.header.frame_id = "map";
        global_occ_grid_msg.info.resolution = resolution;
        global_occ_grid_msg.info.width = map_size;
        global_occ_grid_msg.info.height = map_size;
        global_occ_grid_msg.info.origin.position.x = -map_size * resolution / 2.0+20;
        global_occ_grid_msg.info.origin.position.y = -map_size * resolution / 2.0+10;
        global_occ_grid_msg.data = global_occ_grid;

        semantic_map_pub.publish(global_occ_grid_msg);
    }

    void publishTransform()
    {
        try
        {
            // Create a TransformStamped message
            geometry_msgs::TransformStamped transform;

            // Header
            transform.header.stamp = ros::Time::now();
            transform.header.frame_id = "map";
            transform.child_frame_id = "odom";
            transform.transform.rotation.w = 1.0;

            // Publish the static transform
            tf_broadcaster->sendTransform(transform);
        }
        catch (const ros::Exception &e)
        {
            ROS_ERROR_STREAM("Error while publishing transform: " << e.what());
        }
    }

    bool checkForUpdate()
    {
        double dx = std::sqrt(pow(robot_pose[0] - last_robot_pose[0], 2) + pow(robot_pose[1] - last_robot_pose[1], 2));
        double dtheta = std::abs(robot_pose[2] - last_robot_pose[2]);

        if (dx >= linear_update || dtheta >= angular_update)
        {
            return true;
        }

        return false;
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber odom_sub;
    ros::Subscriber bev_rgb_sub;
    ros::Publisher semantic_map_pub;

    double resolution;
    int map_size;
    std::vector<int8_t> global_occ_grid;

    double linear_update;
    double angular_update;

    std::vector<double> robot_pose;
    std::vector<double> last_robot_pose;

    tf2_ros::TransformBroadcaster *tf_broadcaster;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "local_occ_grid_concatenator_node");

    SemanticMapper local_occ_grid_concatenator;

    ros::spin();

    return 0;
}