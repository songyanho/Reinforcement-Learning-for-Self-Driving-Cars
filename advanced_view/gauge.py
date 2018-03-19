#
# Copyright (c) 2014 Nick Dajda <nick.dajda@gmail.com>
#
# Distributed under the terms of the GNU GENERAL PUBLIC LICENSE
#
"""Python gauge for PIL

    Typical usage:
        im = Images.new(dimensions, colors, ...)
        gauge = gaugeDraw(im, min, max, % of dial) <-- extends ImageDraw
        gauge.add_dial_labels(dictionary) <-- e.g. {0: 'N', 90: 'E', 180: 'S', 270: 'W'}
        gauge.add_needle(value)
        gauge.add_history(list, num_buckets)
        gauge.add_dial(minor_tick, major_tick)
        gauge.add_text( ("27", "degC", "(very hot)") )
        gauge.render()
        im.save("filename for png file")
"""

import math

from PIL import ImageDraw
from PIL import ImageFont

DEFAULT_FONT = "/usr/share/fonts/truetype/freefont/FreeSans.ttf"


class GaugeDraw(ImageDraw.ImageDraw):
    """Class for rendering nice gauge images, e.g. for use on a weather station website."""

    def __init__(self, im, min_val, max_val, dial_range=270, background_color=None, offset_angle=0):
        """Initialises the dial.
           min_val = minimum value on dial
           max_val = maximum value on dial
           dial_range = any value between 0 and 360.
                        360: dial is a complete circle
                        180: dial is a semicircle
                        90: dial is a quarter of a complete circle
            offset_angle = Change the point in the circle that the gauge begins and ends.self
                        0: gauge starts and end around the bottom of the image_height
                        90: the left
                        180: the top - useful for generating a compass gauge"""

        # This class extends ImageDraw... Initialise it
        ImageDraw.ImageDraw.__init__(self, im)

        self.min_value = float(min_val)
        self.max_value = float(max_val)

        if dial_range < 360:
            self.min_angle = (360 - dial_range) / 2
            self.max_angle = 360 - self.min_angle
        else:
            self.min_angle = 0
            self.max_angle = 360

        # Derive image dimensions from im
        (self.image_width, self.image_height) = im.size
        self.gauge_origin = (int(self.image_width / 2), int(self.image_height / 2))

        if self.image_width < self.image_height:
            self.radius = self.image_width * 0.45
        else:
            self.radius = self.image_height * 0.45

        # If None, means no histogram data added
        self.num_buckets = None

        # Whether to draw the dial
        self.draw_dial = False

        # No value set
        self.gauge_value = None

        # Text caption will be stored here
        self.text_labels = None

        # Dial labels
        self.dial_labels = None

        # Default colors...
        self.colors = {'histogram': 4342452,
                       'background': 16777215,
                       'dial_label': 0,
                       'dial': 7368816,
                       'needle_outline': "#e22222",
                       'needle_fill': "#e22222",
                       'text': "#e22222"}

        if background_color is not None:
            self.colors['background'] = background_color

        self.fill_color_tuple = int2rgb(self.colors['histogram'])
        self.back_color_tuple = int2rgb(self.colors['background'])

        self.offset_angle = offset_angle

    def add_needle(self, value, needle_outline_color=None, needle_fill_color=None):
        """Draws a needle pointing towards value.

        needle_outline_color overrides the default"""
        self.gauge_value = value

        if needle_outline_color is not None:
            self.colors['needle_outline'] = needle_outline_color

        if needle_fill_color is not None:
            self.colors['needle_fill'] = needle_fill_color

    def add_dial_labels(self, dial_labels=[], dial_label_font_size=12, dial_label_color=None,
                        dial_label_font=None):
        """Takes a dictionary and draws text at every key.
        On a dial from 0 to 360, this dictionary would print the points of the compoass:
        {0: 'N', 90: 'E', 180: 'S', 270: 'W'}"""
        if type(dial_labels) is dict:
            self.dial_labels = dial_labels

        if dial_label_font is None:
            dial_label_font = DEFAULT_FONT

        self.dial_label_font = get_font(dial_label_font, dial_label_font_size)

        if dial_label_color is not None:
            self.colors['dial_label'] = dial_label_color

    def add_text(self, text_list=None, text_font_size=20,
                 text_font=None, text_color=None):
        """Adds multiple lines of text as a caption.
        Usually used to display the value of the gauge.

        If label_list is not set, will create a single line label based on the value the needle is pointing to
        (only works if add_needle function has already been called)."""

        if text_list is None:
            if self.gauge_value is None:
                # Not enough information to do anything useful
                return
            else:
                text_list = str(self.gauge_value)

        self.text_labels = []

        if type(text_list) is tuple:
            for l in text_list:
                self.text_labels.append(l)
        else:
            self.text_labels.append(text_list)

        if text_font is None:
            text_font = DEFAULT_FONT

        self.text_font = get_font(text_font, text_font_size)
        self.text_font_size = text_font_size

        if text_color is not None:
            self.colors['text'] = text_color

    def add_dial(self, major_ticks, minor_ticks=None, dial_format="%.1f", dial_font_size=12,
                 dial_font=None, dial_color=None, dial_label_color=None, dial_thickness=1):
        """Configures the background dial
        major_ticks and minor_ticks are how often to add a tick mark to the dial.

        Set dial_format to None to stop labelling every major tick mark"""

        try:
            self.major_tick = float(major_ticks)
        except:
            raise Exception("Need to specify a number for major_ticks.")

        self.minor_tick = minor_ticks
        self.dial_format = dial_format

        if dial_font is None:
            dial_font = DEFAULT_FONT

        self.dial_font = get_font(dial_font, dial_font_size)

        if dial_color is not None:
            self.colors['dial'] = dial_color

        if dial_label_color is not None:
            self.colors['dial_label'] = dial_label_color

        self.dial_thickness = dial_thickness

        self.draw_dial = True

    def add_history(self, list_vals, num_buckets, histogram_color=None):
        """Turn list_vals of values into a histogram"""
        if num_buckets is None:
            raise Exception("Need to specify number of buckets to split histogram into.")

        self.num_buckets = num_buckets

        if list_vals is None:
            raise Exception("No data specified.")

        self.buckets = [0.0] * num_buckets
        bucket_span = (self.max_value - self.min_value) / num_buckets
        num_points = 0
        roof = 0.0

        for data in list_vals:
            # Ignore data which is outside range of gauge
            if (data < self.max_value) and (data > self.min_value):
                bucket = int((data - self.min_value) / bucket_span)

                if bucket >= num_buckets:
                    raise Exception("Value %f gives bucket higher than num_buckets (%d)" % (data, num_buckets))
                else:
                    self.buckets[bucket] += 1.0
                    num_points += 1

                    if self.buckets[bucket] > roof:
                        roof = self.buckets[bucket]

        if roof != 0.0:
            self.buckets = [i / roof for i in self.buckets]

            if histogram_color is not None:
                self.colors['histogram'] = histogram_color
                self.fill_color_tuple = int2rgb(self.colors['histogram'])
        else:
            # No history values found within the visible range of the gauge
            self.num_buckets = None

    def render_simple_gauge(self, value=None, major_ticks=None, minor_ticks=None, label=None, font=None):
        """Helper function to create gauges with minimal code, eg:

            import Image
            import gauges

            im = Image.new("RGB", (200, 200), (255, 255, 255))
            g = gauges.GaugeDraw(im, 0, 100)
            g.render_simple_gauge(value=25, major_ticks=10, minor_ticks=2, label="25")
            im.save("simple_gauge_image.png", "PNG")

        Does not support dial labels, histogram dial background or setting colors..
        """
        if value is not None:
            self.add_needle(value)

        if major_ticks is not None:
            self.add_dial(major_ticks, minor_ticks, dial_font=font)

        if label is not None:
            self.add_text(text_list=label, text_font=font)

        self.render()

    def draw_buckets(self):
        """Draws the history buckets."""
        if self.num_buckets is not None:
            angle = float(self.min_angle)
            angle_step = (self.max_angle - self.min_angle) / float(self.num_buckets)

            for bucket in self.buckets:
                fill_color = (self._calc_color(bucket, 0), self._calc_color(bucket, 1), self._calc_color(bucket, 2))

                self.pieslice((int(self.gauge_origin[0] - self.radius), int(self.gauge_origin[1] - self.radius),
                               int(self.gauge_origin[0] + self.radius), int(self.gauge_origin[1] + self.radius)),
                              int(angle + 90 + self.offset_angle), int(angle + angle_step + 90 + self.offset_angle),
                              fill=fill_color)
                angle += angle_step

    def draw_scale(self):
        """Draws the dial with tick marks and dial labels"""
        if self.draw_dial is True:
            # Major tic marks and scale labels
            label_value = self.min_value

            for angle in self._frange(math.radians(self.min_angle + self.offset_angle),
                                      math.radians(self.max_angle + self.offset_angle),
                                      int(1 + (self.max_value - self.min_value) / self.major_tick)):

                start_point = (self.gauge_origin[0] - self.radius * math.sin(angle)
                               * 0.93, self.gauge_origin[1] + self.radius * math.cos(angle) * 0.93)

                end_point = (self.gauge_origin[0] - self.radius * math.sin(angle),
                             self.gauge_origin[1] + self.radius * math.cos(angle))

                self._thick_line(start_point, end_point, fill=self.colors['dial'], thickness=self.dial_thickness)

                if self.dial_format is not None and self.dial_format != 'None':
                    text = str(self.dial_format % label_value)
                    string_size = self.dial_font.getsize(text)

                    label_point = (self.gauge_origin[0] - self.radius * math.sin(angle) * 0.80,
                                   self.gauge_origin[1] + self.radius * math.cos(angle) * 0.80)

                    label_point = (label_point[0] - string_size[0] / 2, label_point[1] - string_size[1] / 2)

                    self.text(label_point, text, font=self.dial_font, fill=self.colors['dial_label'])

                    label_value += self.major_tick

            # Minor tic marks
            if self.minor_tick is not None:
                for angle in self._frange(math.radians(self.min_angle + self.offset_angle),
                                          math.radians(self.max_angle + self.offset_angle),
                                          int(1 + (self.max_value - self.min_value) / self.minor_tick)):
                    start_point = (self.gauge_origin[0] - self.radius * math.sin(angle) * 0.97,
                                   self.gauge_origin[1] + self.radius * math.cos(angle) * 0.97)

                    end_point = (self.gauge_origin[0] - self.radius * math.sin(angle),
                                 self.gauge_origin[1] + self.radius * math.cos(angle))

                    self._thick_line(start_point, end_point, fill=self.colors['dial'], thickness=self.dial_thickness)

            # The edge of the dial
            self._thick_arc((self.gauge_origin[0] - int(self.radius), self.gauge_origin[1] - int(self.radius),
                             self.gauge_origin[0] + int(self.radius), self.gauge_origin[1] + int(self.radius)),
                            self.min_angle + 90 + self.offset_angle, self.max_angle + 90 + self.offset_angle,
                            self.colors['dial'], thickness=self.dial_thickness)

            # Custom gauge labels?
            if self.dial_labels is not None:
                for k in self.dial_labels.keys():
                    angle = (k - self.min_value) / (self.max_value - self.min_value)
                    if (angle >= 0.0) and (angle <= 1):
                        angle = math.radians(self.min_angle + angle * (self.max_angle - self.min_angle)
                                             + self.offset_angle)

                        string_size = self.dial_label_font.getsize(self.dial_labels[k])

                        label_point = (self.gauge_origin[0] - self.radius * math.sin(angle) * 0.80,
                                       self.gauge_origin[1] + self.radius * math.cos(angle) * 0.80)

                        label_point = (label_point[0] - string_size[0] / 2, label_point[1] - string_size[1] / 2)

                        self.text(label_point, self.dial_labels[k], font=self.dial_label_font,
                                  fill=self.colors['dial_label'])

    def draw_labels(self):
        """Draws the reading/text label"""
        if self.text_labels is not None:
            vstep = self.text_font_size * 1.3
            vpos = self.gauge_origin[1] + self.radius * 0.42 - (vstep * len(self.text_labels)) / 2

            for l in self.text_labels:
                text = unicode(l.encode("utf-8"), 'utf8')
                textsize = self.text_font.getsize(text)

                self.text((self.gauge_origin[0] - (textsize[0] / 2), vpos), text,
                          font=self.text_font, fill=self.colors['text'])
                vpos += vstep

    def draw_needle(self):
        """Draws the needle"""
        if self.gauge_value is not None:

            if self.gauge_value < self.min_value:
                self.gauge_value = self.min_value

            if self.gauge_value > self.max_value:
                self.gauge_value = self.max_value

            angle = math.radians(self.min_angle + (self.gauge_value - self.min_value) *
                                 (self.max_angle - self.min_angle) / (self.max_value - self.min_value)
                                 + self.offset_angle)

            end_point = (self.gauge_origin[0] - self.radius * math.sin(angle) * 0.7, self.gauge_origin[1]
                         + self.radius * math.cos(angle) * 0.7)
            left_point = (self.gauge_origin[0] - self.radius * math.sin(angle - math.pi * 7 / 8) * 0.2,
                          self.gauge_origin[1] + self.radius * math.cos(angle - math.pi * 7 / 8) * 0.2)
            right_point = (self.gauge_origin[0] - self.radius * math.sin(angle + math.pi * 7 / 8) * 0.2,
                           self.gauge_origin[1] + self.radius * math.cos(angle + math.pi * 7 / 8) * 0.2)
            mid_point = (self.gauge_origin[0] - self.radius * math.sin(angle + math.pi) * 0.1,
                         self.gauge_origin[1] + self.radius * math.cos(angle + math.pi) * 0.1)

            self._thick_polygon((left_point, end_point, right_point, mid_point), outline=self.colors['needle_outline'],
                                fill=self.colors['needle_fill'], thickness=self.dial_thickness)

    def render(self):
        """Renders the gauge. Call this function last."""
        self.draw_buckets()  # History
        self.draw_scale()  # Dial and dial labels
        self.draw_labels()  # Reading/Text labels
        self.draw_needle()  # Do last - the needle is on top of everything

    @staticmethod
    def _frange(start, stop, n):
        """Range function, for floating point numbers"""
        l = [0.0] * n
        nm1 = n - 1
        nm1inv = 1.0 / nm1
        for i in range(n):
            l[i] = nm1inv * (start * (nm1 - i) + stop * i)
        return l

    def _thick_polygon(self, points, outline=None, fill=None, thickness=1):
        """Draws a polygon outline using polygons to give it a thickness"""
        if thickness == 1:
            self.polygon(points, outline=outline, fill=fill)
        else:
            if fill is not None:
                # Fill it in before drawing the exterior
                self.polygon(points, outline=outline, fill=fill)

            # Outline needed?
            if outline is not None:
                last_point = None

                for point in points:
                    if last_point is not None:
                        self._thick_line(last_point, point, fill=outline, thickness=thickness)

                    last_point = point

                self._thick_line(point, points[0], fill=outline, thickness=thickness)

    def _thick_arc(self, bbox, start, end, fill, thickness):
        """Draws an arc using polygons to give it a thickness"""
        num_segments = 50

        if thickness == 1:
            self.arc(bbox, start, end, fill=fill)
        else:
            start *= (math.pi / 180)
            end *= (math.pi / 180)

            rx = (bbox[2] - bbox[0]) / 2.0
            ry = (bbox[3] - bbox[1]) / 2.0

            midx = (bbox[2] + bbox[0]) / 2.0
            midy = (bbox[3] + bbox[1]) / 2.0

            angle_step = (end - start) / num_segments

            for angle in self._frange(start, end, num_segments):
                end_angle = angle + angle_step
                if end_angle > end:
                    end_angle = end

                x1 = midx + rx * math.cos(angle)
                y1 = midy + ry * math.sin(angle)

                x2 = midx + rx * math.cos(end_angle)
                y2 = midy + ry * math.sin(end_angle)

                self._thick_line((x1, y1), (x2, y2), fill, thickness)

    def _thick_line(self, start_point, end_point, fill, thickness):
        """Draws a line using polygons to give it a thickness"""

        if thickness == 1:
            self.line((start_point, end_point), fill=fill)
        else:
            # Angle of the line
            if end_point[0] == start_point[0]:
                # Catch a division by zero error
                a = math.pi / 2
            else:
                a = math.atan((end_point[1] - start_point[1]) / (end_point[0] - start_point[0]))

            sin = math.sin(a)
            cos = math.cos(a)
            xdelta = sin * thickness / 2.0
            ydelta = cos * thickness / 2.0

            points = ((start_point[0] - xdelta, start_point[1] + ydelta),
                      (start_point[0] + xdelta, start_point[1] - ydelta),
                      (end_point[0] + xdelta, end_point[1] - ydelta),
                      (end_point[0] - xdelta, end_point[1] + ydelta))

            self.polygon(points, fill=fill)

    def _calc_color(self, value, index):
        diff = self.fill_color_tuple[index] - self.back_color_tuple[index]
        new_color = self.back_color_tuple[index] + int(diff * value)

        if new_color < 0:
            new_color = 0

        if new_color > 0xff:
            new_color = 0xff

        return new_color


class WindRoseGaugeDraw(GaugeDraw):
    """Class for rendering a meteorological wind rose"""

    def __init__(self, im, background_color=None):
        """Initialises the dial.
            background_color = color outside the dial"""

        # This class extends GaugeDraw... Initialise it
        GaugeDraw.__init__(self, im, 0, 360, dial_range=360, background_color=background_color, offset_angle=180)

    def add_history(self, list_vals, num_buckets, ring_vals=None, rings=None, ring_colors=None):
        """Turn list_vals of values into a histogram

        Polar history data get mapped to polar coordinates. Angular dimension are Vals, distance dimension is number of data point in per angular bucket.
        Buckets can be divided into rings. Values are mapped to rings via rings.

        Ring 0 does not get drawn. If you want to have one, put a lower limit in rings.

        list_vals = angular values, assigned to buckets by dividing 360 degree by bucket_num. Typical wind direction.
        ring_vals = List of values for rings. Typical wind speed ranges.
        rings = Mapping instruction for ring values
        ring_colors = Colors for the rings"""

        if num_buckets is None:
            raise Exception("Need to specify number of buckets to split histogram into.")

        self.num_buckets = num_buckets

        if list_vals is None:
            raise Exception("No data specified.")

        self.num_rings = 0

        if ring_vals is not None:
            if rings is None:
                raise Exception("No ring ranges specified.")
            if len(ring_vals) != len(list_vals):
                raise Exception("Number of ring vals (%d) does not match the number of list vals (%d)." % (
                len(ring_vals), len(list_vals)))
            if len(rings) > len(ring_colors):
                raise Exception("Number of ring colors (%d) does not match the number of rings (%d)." % (
                len(ring_colors), len(rings)))
            self.num_rings = len(rings)
            self.ring_colors = ring_colors

        # Buckets contains a list of buckets.
        # A bucket is list which in turn contains at index [0] the number of data points and at index[1]
        # a list of rings
        # The list of rings contains for each ring the number of data point
        self.buckets = []
        for i in range(num_buckets):
            self.buckets.append([0.0, [0.0] * self.num_rings])

        bucket_span = (self.max_value - self.min_value) / num_buckets

        for i in range(len(list_vals)):
            data = list_vals[i]
            # Ignore data which is outside range of gauge
            if (data > self.min_value) and (data < self.max_value):
                data = (data + 360 + (bucket_span / 2)) % self.max_value  # take bucket offset into account
                bucket = int((data - self.min_value) / bucket_span)

                if bucket >= num_buckets:
                    raise Exception("Value %f gives bucket higher than num_buckets (%d)" % (data, num_buckets))
                else:
                    self.buckets[bucket][0] += 1.0
                    if ring_vals is not None:
                        ring = get_ring(ring_vals[i], rings)
                        self.buckets[bucket][1][ring] += 1.0

        bucket_max = max(self.buckets)[0]
        if abs(bucket_max) > 0:
            for bucket in self.buckets:
                bucket[0] /= bucket_max
                ring_sum = max(1.0, sum(bucket[1][1:]))
                ring = 0.0
                for k in range(1, len(bucket[1])):
                    if bucket[1][k] != 0:
                        ring += bucket[1][k] / ring_sum
                        bucket[1][k] = ring

    def draw_buckets(self):
        """Draw the wind rose.
            - Bucket size is relative number of entries in buckets
            - Bucket color shade is absolute wind speed in beaufort"""
        if self.num_buckets is not None:
            angle = float(self.min_angle)
            angle_step = (self.max_angle - self.min_angle) / float(self.num_buckets)
            bucket_angle_offset = angle_step / float(2)

            for bucket in self.buckets:
                for i in reversed(range(1, len(bucket[1]))):
                    ring = bucket[1][i]
                    radius = self.radius * bucket[0] * ring
                    if radius > 0:
                        self.pieslice((int(self.gauge_origin[0] - radius), int(self.gauge_origin[1] - radius),
                                       # bounding box x0, y0
                                       int(self.gauge_origin[0] + radius), int(self.gauge_origin[1] + radius)),
                                      # bounding box x1, y1
                                      int(angle + 90 + self.offset_angle - bucket_angle_offset),  # start angle
                                      int(angle + angle_step + 90 + self.offset_angle - bucket_angle_offset),
                                      # end angle
                                      outline=self.colors["dial"],
                                      fill=self.ring_colors[i])
                angle += angle_step


def get_font(font_path, font_size):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    return font


def get_ring(value, rings):
    for i in range(len(rings)):
        if value < rings[i]:
            return i
    return len(rings)


def int2rgb(x):
    #
    # Stolen from genploy.py Weewx file
    #
    b = (x >> 16) & 0xff
    g = (x >> 8) & 0xff
    r = x & 0xff
    return r, g, b