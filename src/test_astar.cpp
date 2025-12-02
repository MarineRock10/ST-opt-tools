/*
    MIT License

    Copyright (c) 2025 Senming Tan (senmingtan5@gmail.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#include "astar.hpp"
#include "grid_map.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/Dense>

// Visualize A* planning results with OpenCV
void visualizeTrajectory(
    const grid_map::GridMap& map,
    const path_planning::AStar::Trajectory& traj,
    const Eigen::Vector2d& start,
    const Eigen::Vector2d& goal,
    double viz_size = 30.0
) {
    const double resolution = map.getResolution();
    const int scale = 4; // Scaling factor for visualization
    const int viz_pixels = static_cast<int>(viz_size / resolution) * scale;
    
    // Create visualization image
    cv::Mat viz_img(viz_pixels, viz_pixels, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Coordinate conversion (world to pixel)
    auto worldToPixel = [&](const Eigen::Vector2d& pt) {
        int x = viz_pixels/2 + pt.x() / resolution * scale;
        int y = viz_pixels/2 - pt.y() / resolution * scale; // Flip Y-axis
        return cv::Point(x, y);
    };

    // 1. Draw map boundary (red)
    Eigen::Vector2d map_min = -map.getMapSize()/2.0;
    Eigen::Vector2d map_max = map.getMapSize()/2.0;
    cv::rectangle(viz_img, worldToPixel(map_min), worldToPixel(map_max), 
                 cv::Scalar(0, 0, 255), 2);

    // 2. Draw obstacles (black)
    for (int x = 0; x < map.getVoxelNum().x(); ++x) {
        for (int y = 0; y < map.getVoxelNum().y(); ++y) {
            if (map.isOccupied(Eigen::Vector2i(x, y))) {
                Eigen::Vector2d pos;
                map.indexToPos(Eigen::Vector2i(x, y), pos);
                cv::rectangle(viz_img, 
                    worldToPixel(pos) - cv::Point(scale/2, scale/2),
                    worldToPixel(pos) + cv::Point(scale/2, scale/2), 
                    cv::Scalar(0, 0, 0), -1);
            }
        }
    }

    // 3. Draw raw path (blue)
    for (const auto& pt : traj.raw_path) {
        cv::circle(viz_img, worldToPixel(pt), scale/3, cv::Scalar(255, 0, 0), -1);
    }

    // 4. Draw optimized path (green)
    for (const auto& pt : traj.optimized_path) {
        cv::circle(viz_img, worldToPixel(pt), scale/2, cv::Scalar(0, 255, 0), -1);
    }

    // 5. Draw state samples (with orientation arrows)
    for (const auto& state : traj.path_states) {
        cv::circle(viz_img, worldToPixel(state.position), scale/4, cv::Scalar(0, 165, 255), -1);
        // Draw orientation arrow
        Eigen::Vector2d end = state.position + Eigen::Vector2d(
            cos(state.theta), sin(state.theta)) * resolution * 3;
        cv::arrowedLine(viz_img, worldToPixel(state.position), worldToPixel(end),
                       cv::Scalar(0, 165, 255), 1, cv::LINE_AA, 0, 0.2);
    }

    // 6. Draw timed trajectory (color gradient shows time progression)
    if (!traj.timed_trajectory.empty()) {
        double max_time = traj.total_time;
        for (const auto& point : traj.timed_trajectory) {
            double ratio = point.time / max_time;
            cv::circle(viz_img, worldToPixel(point.state.head<2>()), scale/5, 
                      cv::Scalar(255 * ratio, 0, 255 * (1 - ratio)), -1);
        }
    }

    // 7. Mark start and goal points
    cv::circle(viz_img, worldToPixel(start), scale, cv::Scalar(0, 0, 255), -1);
    cv::circle(viz_img, worldToPixel(goal), scale, cv::Scalar(255, 0, 0), -1);
    cv::putText(viz_img, "Start", worldToPixel(start) + cv::Point(scale, 0), 
              cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
    cv::putText(viz_img, "Goal", worldToPixel(goal) + cv::Point(scale, 0), 
              cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);

    // Add legend
    cv::putText(viz_img, "Raw Path (Blue)", cv::Point(10, 20), 
              cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
    cv::putText(viz_img, "Optimized Path (Green)", cv::Point(10, 40), 
              cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
    cv::putText(viz_img, "States (Orange)", cv::Point(10, 60), 
              cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 165, 255), 1);
    cv::putText(viz_img, "Timed Traj (Purple->Red)", cv::Point(10, 80), 
              cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 255), 1);

    // Display image
    cv::imshow("A* Path Planning with Post-processing", viz_img);
    cv::waitKey(0);
}

int main() {
    // ==================== 1. Initialize local map (20x20m) ====================
    grid_map::GridMap map;
    const double map_size = 20.0;
    const double resolution = 0.1;
    map.init(map_size, map_size, resolution);
    
    // ==================== 2. Set obstacles ====================
    grid_map::RowMatrixXi occupancy = grid_map::RowMatrixXi::Zero(200, 200);
    // Horizontal wall (x: -5m~5m, y: 0m)
    for (int x = 50; x < 150; ++x) {
        for (int y = 95; y < 105; ++y) {
            occupancy(x, y) = 1;
        }
    }
    // Vertical wall (x: -8m~-3m, y: 5m~10m)
    for (int x = 20; x < 70; ++x) {
        for (int y = 150; y < 200; ++y) {
            occupancy(x, y) = 1;
        }
    }
    map.setMap(occupancy);

    // ==================== 3. Define start and goal ====================
    const Eigen::Vector2d start(-8.0, -8.0);  // Bottom-left corner
    const Eigen::Vector2d goal(8.0, 8.0);     // Top-right corner

    // ==================== 4. Execute A* planning with post-processing ====================
    path_planning::AStar planner(map, 0.3); // 0.3m safety threshold
    
    // Set planning parameters
    planner.setMaxVelocity(2.0);
    planner.setMaxAcceleration(1.0);
    planner.setTimeResolution(0.2);
    planner.setMinTrajectoryNumber(10);
    
    auto traj = planner.planWithPostProcessing(start, goal, 5000); // 5s timeout
    
    // ==================== 5. Output results ====================
    if (!traj.raw_path.empty()) {
        std::cout << "=== Planning Results ===" << std::endl;
        std::cout << "Raw path points: " << traj.raw_path.size() << std::endl;
        std::cout << "Optimized path points: " << traj.optimized_path.size() << std::endl;
        std::cout << "Total length: " << traj.total_length << " m" << std::endl;
        std::cout << "Total time: " << traj.total_time << " s" << std::endl;
        
        // Print key points
        std::cout << "\nKey points in optimized path:" << std::endl;
        for (size_t i = 0; i < traj.optimized_path.size(); i += traj.optimized_path.size()/5) {
            std::cout << "Point " << i << ": (" 
                     << traj.optimized_path[i].x() << ", " 
                     << traj.optimized_path[i].y() << ")" << std::endl;
        }
        
        std::cout << "\nTimed trajectory samples:" << std::endl;
        for (size_t i = 0; i < traj.timed_trajectory.size(); i += traj.timed_trajectory.size()/5) {
            std::cout << "Time " << traj.timed_trajectory[i].time << " s: ("
                     << traj.timed_trajectory[i].state.x() << ", "
                     << traj.timed_trajectory[i].state.y() << ", "
                     << traj.timed_trajectory[i].state.z() << " rad)" << std::endl;
        }
    } else {
        std::cerr << "Planning failed!" << std::endl;
        return -1;
    }

    // ==================== 6. Visualization ====================
    visualizeTrajectory(map, traj, start, goal, 25.0);

    return 0;
}