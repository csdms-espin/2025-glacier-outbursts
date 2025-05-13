from landlab.components import (
    LinearDiffuser, OverlandFlow, FlowDirectorSteepest, FlowAccumulator,
    SedDepEroder, ChannelProfiler, PriorityFloodFlowRouter, DepressionFinderAndRouter, Space
)
from landlab import RasterModelGrid, imshow_grid
from landlab.plot.graph import plot_graph
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

#create ThawIce function #cred: Lucille
class ThawIce:
    #define function and input
    def __init__ (self, ice_thickness):
        self.ice_thickness = ice_thickness
    #define thaw function and inputs
    def thaw (self, thaw_rate, dt):
        self.thaw_rate = thaw_rate
        self.dt = dt
        for i in range(len(self.ice_thickness)):
            if self.ice_thickness[i] >= self.thaw_rate*self.dt:
                self.ice_thickness[i] = self.ice_thickness[i] - self.dt*self.thaw_rate
            else:
                self.ice_thickness[i] = 0
            
# Method 1: Parabolic circular mounds
def create_parabolic_mound(x_center, y_center, radius, depth_peak):
    """Create a parabolic soil mound with specified radius and peak depth"""
    # Calculate distance from center for all nodes
    distance = np.sqrt((grid.x_of_node - x_center)**2 + (grid.y_of_node - y_center)**2)
    # Create parabolic profile: depth = peak * (1 - (distance/radius)^2)
    # Only apply within the specified radius
    normalized_distance = distance / radius
    # Create mask for nodes within the mound
    within_mound = distance <= radius
    # Calculate parabolic depth (only for nodes within radius)
    parabolic_factor = np.zeros_like(distance)
    parabolic_factor[within_mound] = 1 - normalized_distance[within_mound]**2
    # Ensure no negative values
    parabolic_factor = np.maximum(parabolic_factor, 0)
    # Calculate depth to add
    depth_to_add = depth_peak * parabolic_factor
    # Add the new depth to existing soil depth
    grid.at_node["soil__depth"] += depth_to_add
    return depth_to_add
#define the grid
size_x = 100
size_y = 200
spacing = 1.0
grid = RasterModelGrid((size_y, size_x), xy_spacing=spacing)
# morph the grid with a sine function
ice_height = 10
bedrock_slope = 0.22
soil_thickness = 5
moraine_disappears_at_y = 100
# add the topography
z_ice = grid.add_zeros('ice_thickness', at='node')
z_soil = grid.add_zeros('soil__depth', at='node')
z_bed = grid.add_zeros('bedrock__elevation', at='node')
elev = grid.add_zeros('topographic__elevation', at='node')
#base_soil_depth = 1  # meters
grid.at_node["soil__depth"][:] = soil_thickness
# Large central parabolic mound
create_parabolic_mound(x_center=25, y_center=25, radius=10, depth_peak=10)
# Set bedrock elevation equal to the initial topography
grid.at_node["bedrock__elevation"][:] = grid.at_node["topographic__elevation"]
# Update topography to include soil depth (topography = bedrock + soil)
grid.at_node["topographic__elevation"][:] = grid.at_node["bedrock__elevation"] + grid.at_node["soil__depth"]
x = grid.x_of_node
y = grid.y_of_node
# setting ice elevation respected to the base elevation
z = ice_height * np.sin(np.pi*x / (size_x/3))
# get z where y is less than moraine_disappears_at_y
z[y<moraine_disappears_at_y] += (y[y<moraine_disappears_at_y] - moraine_disappears_at_y) * ice_height/moraine_disappears_at_y
z[z<0] = 0 # cut the sine function at 0
z_ice += z
z_soil += soil_thickness
z_bed += bedrock_slope * y
elev = z_ice + z_soil + z_bed
# reassign the values to the grid just to be sure
grid.at_node['ice_thickness'] = z_ice
grid.at_node['soil_thickness'] = z_soil
grid.at_node['bedrock_thickness'] = z_bed
grid.at_node['topographic__elevation'] = elev
grid.imshow(grid.at_node["topographic__elevation"], color_for_closed = 'm')
plt.show()
# 3D visualization of topography
ax_3d = fig.add_subplot(2, 3, 6, projection='3d')
x = grid.x_of_node.reshape(grid.shape)
y = grid.y_of_node.reshape(grid.shape)
z_topo = grid.at_node["topographic__elevation"].reshape(grid.shape)
surf = ax_3d.plot_surface(x, y, z_topo, cmap='terrain', edgecolor='none',
                         linewidth=0, antialiased=True, alpha=0.8)
ax_3d.set_xlabel('X (m)')
ax_3d.set_ylabel('Y (m)')
ax_3d.set_zlabel('Elevation (m)')
ax_3d.set_title('3D Topography with Parabolic Mounds')
ax_3d.view_init(elev=25, azim=225)
plt.tight_layout()
plt.show()
# Visualize the results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
# Top row - main views
# Plot soil depth
plt.sca(axes[0, 0])
soil_map = grid.imshow("soil__depth", color_for_closed='m', cmap='viridis')
axes[0, 0].set_title('Soil Depth')
plt.colorbar(soil_map, label='Soil Depth (m)')
# Plot bedrock elevation
plt.sca(axes[0, 1])
bedrock_map = grid.imshow("bedrock__elevation", color_for_closed='m', cmap='terrain')
axes[0, 1].set_title('Bedrock Elevation')
plt.colorbar(bedrock_map, label='Bedrock Elevation (m)')
# Plot topographic elevation
plt.sca(axes[0, 2])
topo_map = grid.imshow("topographic__elevation", color_for_closed='m', cmap='terrain')
axes[0, 2].set_title('Topographic Elevation')
plt.colorbar(topo_map, label='Topographic Elevation (m)')
# Bottom row - cross-sections to show mound profiles
# Cross-section through central mound (horizontal)
y_section = 25
nodes_in_section = grid.nodes[:, y_section].flatten()
x_coords = grid.x_of_node[nodes_in_section]
soil_depth_section = grid.at_node["soil__depth"][nodes_in_section]
topo_section = grid.at_node["topographic__elevation"][nodes_in_section]
axes[1, 0].plot(x_coords, soil_depth_section, 'b-', linewidth=2, label='Soil Depth')
axes[1, 0].set_xlabel('X (m)')
axes[1, 0].set_ylabel('Soil Depth (m)')
axes[1, 0].set_title(f'Soil Depth Cross-section (y={y_section})')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()
# Cross-section through central mound (vertical)
x_section = 25
y_indices = np.arange(n_rows)
nodes_vertical = []
for j in y_indices:
    nodes_vertical.append(grid.nodes[j, x_section])
nodes_vertical = np.array(nodes_vertical)
y_coords = grid.y_of_node[nodes_vertical]
soil_depth_vertical = grid.at_node["soil__depth"][nodes_vertical]
axes[1, 1].plot(y_coords, soil_depth_vertical, 'r-', linewidth=2, label='Soil Depth')
axes[1, 1].set_xlabel('Y (m)')
axes[1, 1].set_ylabel('Soil Depth (m)')
axes[1, 1].set_title(f'Soil Depth Cross-section (x={x_section})')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()
# 3D visualization of topography
ax_3d = fig.add_subplot(2, 3, 6, projection='3d')
x = grid.x_of_node.reshape(grid.shape)
y = grid.y_of_node.reshape(grid.shape)
z_topo = grid.at_node["topographic__elevation"].reshape(grid.shape)
surf = ax_3d.plot_surface(x, y, z_topo, cmap='terrain', edgecolor='none',
                         linewidth=0, antialiased=True, alpha=0.8)
ax_3d.set_xlabel('X (m)')
ax_3d.set_ylabel('Y (m)')
ax_3d.set_zlabel('Elevation (m)')
ax_3d.set_title('3D Topography with Parabolic Mounds')
ax_3d.view_init(elev=25, azim=225)
plt.tight_layout()
plt.show()