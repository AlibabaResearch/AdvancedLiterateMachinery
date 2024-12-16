import math
colors = [
    (255, 255, 255), (230, 230, 230), (200, 200, 200),
    (150, 150, 150), (100, 100, 100), (50, 50, 50),
    (0, 0, 0), (255, 0, 0), (255, 153, 0),
    (255, 255, 0), (153, 255, 0), (0, 255, 0),
    (0, 255, 153), (0, 255, 255), (0, 153, 255),
    (0, 0, 255), (153, 0, 255), (255, 0, 255),
    (255, 153, 204), (255, 204, 204), (255, 255, 204),
    (204, 255, 204), (204, 255, 255), (204, 204, 255),
    (255, 204, 255), (204, 153, 0), (255, 102, 0), 
    (255, 204, 0), (153, 204, 0), (51, 204, 51),
    (0, 204, 153), (51, 153, 255), (204, 153, 255),
    (255, 102, 204), (204, 0, 0), (204, 102, 0),
    (204, 204, 0), (102, 204, 0), (0, 204, 0),
    (0, 204, 102), (0, 102, 204), (102, 102, 255),
    (204, 0, 204), (102, 0, 0), (102, 51, 0)
]

def rgba_to_rgb(rgba):
    r = int(rgba[0])
    g = int(rgba[1])
    b = int(rgba[2])
    a = float(rgba[3])
    return (int((1 - a) * 255 + a * r), int((1 - a) * 255 + a * g), int((1 - a) * 255 + a * b))

def closest_color(requested_color):
    min_colors = {}
    for key in colors:
        r_c = key[0]
        g_c = key[1]
        b_c = key[2]
        color_diff = math.sqrt((r_c - requested_color[0]) **2 + (g_c - requested_color[1]) **2 + (b_c - requested_color[2]) **2)
        min_colors[color_diff] = key
    return min_colors[min(min_colors.keys())]

def procee_color(value):
    rgba = value.strip().split('(')[1].split(')')[0].split(',')
    if float(rgba[3]) == 0:
        return len(colors)
    rgb = rgba_to_rgb(rgba)  
    rgb = closest_color(rgb)
    return colors.index(rgb)