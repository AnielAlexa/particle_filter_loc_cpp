#include <rclcpp/rclcpp.hpp>
#include "particle_filter_loc_cpp/pf_node.hpp"

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<pf::PFGeoLocNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
