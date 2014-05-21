import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mpl_toolkits.basemap as basemap

regions = {
    'Mississippi': 0,
    'Alabama': 0,
    'Tennessee': 1,
    'Kentucky': 1,
    'Illinois': 2,
    'Indiana': 2,
    'Ohio': 2,
    'Michigan': 3,
    'Wisconsin': 3,
    'Florida': 4,
    'Georgia': 5,
    'South Carolina': 6,
    'North Carolina': 6,
    'Virginia': 7,
    'West Virginia': 7,
    'Maryland': 7,
    'Delaware': 7,
    'Pennsylvania': 8,
    'New Jersey': 8,
    'New York': 8,
    'Connecticut': 9,
    'Rhode Island': 9,
    'Massachusetts': 9,
    'Vermont': 9,
    'New Hampshire': 9,
    'Maine': 9
}
vectors = {
    (0, 1): (-87, 35, 60),
    (1, 2): (-87, 38, 90),
    (2, 3): (-87, 42, 90),
    (0, 4): (-86, 31, -30),
    (0, 5): (-85, 33, 0),
    (4, 5): (-83, 31, 120),
    (1, 6): (-82, 36, -30),
    (5, 6): (-82, 34, 30),
    (1, 7): (-82, 38, 30),
    (2, 7): (-82, 39, -30),
    (6, 7): (-80, 37, 90),
    (2, 8): (-80, 41, 0),
    (7, 8): (-78, 40, 60),
    (8, 9): (-73, 43, 30),
}

def plot_map(values, vmin, vmax, flows):
    m = basemap.Basemap(projection='stere', resolution='c',
                        llcrnrlon=-96, llcrnrlat=24,
                        urcrnrlon=-60, urcrnrlat=48,
                        lon_0=-80, lat_0=36)
    m.drawcoastlines()
    m.drawmapboundary(fill_color="#99ffff")
    m.fillcontinents(color="#ffcc99", lake_color="#99ffff")
    m.drawparallels(np.arange(0, 90, 10), labels=[1,0,0,0])
    m.drawmeridians(np.arange(180, 360, 10), labels=[0,0,0,1])
    m.readshapefile('states/states', 'states', drawbounds=True)
    cmap = plt.cm.hot
    for seg, shapedict in zip(m.states, m.states_info):
        statename = shapedict['STATE_NAME']
        if statename not in regions: continue
        cell = regions[statename]
        val = values[cell]
        color = cmap(1-1.*(val-vmin)/(vmax-vmin))[:3]
        color = colors.rgb2hex(color)
        xx, yy = zip(*seg)
        plt.fill(xx, yy, color, edgecolor=color)
    xs = []
    for from_cell, to_cell, num_birds in flows:
        if vectors.has_key((from_cell, to_cell)):
            lon, lat, angle = vectors[(from_cell, to_cell)]
            xs.append((lon, lat, angle, num_birds))
        elif vectors.has_key((to_cell, from_cell)):
            lon, lat, angle = vectors[(to_cell, from_cell)]
            xs.append((lon, lat, angle, -num_birds))
    lons, lats, angles, nums_birds = np.transpose(xs)
    u = np.cos(angles*np.pi/180) * np.log1p(np.abs(nums_birds)) * np.sign(nums_birds)
    v = np.sin(angles*np.pi/180) * np.log1p(np.abs(nums_birds)) * np.sign(nums_birds)
    m.quiver(lons, lats, u, v, latlon=True,
             pivot='middle', zorder=2, color='#3333ff')
    plt.show()
