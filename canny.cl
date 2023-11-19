typedef uchar uint8_t;
typedef signed short int16_t;

#define PI 3.14159

inline int idx(int x, int y, int width, int height, int xoff, int yoff) {
    int resx = x;
    if ((xoff > 0 && x < width - xoff) || (xoff < 0 && x >= (-xoff)))
        resx += xoff;
    int resy = y;
    if ((yoff > 0 && y < height - yoff) || (yoff < 0 && y >= (-yoff)))
        resy += yoff;
    return resy * width + resx;
}

__kernel void sobel(__global uint8_t* input,
                    __global int16_t* output_x,
                    __global int16_t* output_y,
                    int width, int height) {

   const int x = get_global_id(0); 
   const int y = get_global_id(1);

   int gid = y * width + x;

   /* 3x3 sobel filter, first in x direction */
   output_x[gid] = (-1) * input[idx(x, y, width, height, -1, -1)] +
                  1 * input[idx(x, y, width, height, 1, -1)] +
                  (-2) * input[idx(x, y, width, height, -1, 0)] +
                  2 * input[idx(x, y, width, height, 1, 0)] +
                  (-1) * input[idx(x, y, width, height, -1, 1)] +
                  1 * input[idx(x, y, width, height, 1, 1)];

   /* 3x3 sobel filter, in y direction */
   output_y[gid] = (-1) * input[idx(x, y, width, height, -1, -1)] +
                  1 * input[idx(x, y, width, height, -1, 1)] +
                  (-2) * input[idx(x, y, width, height, 0, -1)] +
                  2 * input[idx(x, y, width, height, 0, 1)] +
                  (-1) * input[idx(x, y, width, height, 1, -1)] +
                  1 * input[idx(x, y, width, height, 1, 1)];

}

__kernel void PnM(__global int16_t* input_x,
                    __global int16_t* input_y,
                    __global uint8_t* output_phase,
                    __global int16_t* output_magnitude,
                    int width, int height) {

   const int x = get_global_id(0);
   const int y = get_global_id(1);
   int gid = y * width + x;

      // Output in range -PI:PI
   float angle = atan2((float)input_y[gid], (float)input_x[gid]);

   // Shift range -1:1
   angle /= PI;

   // Shift range -127.5:127.5
   angle *= 127.5;

   // Shift range 0.5:255.5
   angle += (127.5 + 0.5);

   // Downcasting truncates angle to range 0:255
   output_phase[gid] = (uint8_t)angle;

   output_magnitude[gid] = abs(input_x[gid]) + abs(input_y[gid]);

}

__kernel void nonMax(__global uint8_t* input_phase,
                    __global int16_t* input_magnitude,
                    __global uint8_t* output,
                    int width, int height,
                    int threshold_lower,
                    int threshold_upper) {

   int x = get_global_id(0);
   int y = get_global_id(1);

   int gid = y * width + x;

   uint8_t sobel_angle = input_phase[gid];

   if (sobel_angle > 127) {
         sobel_angle -= 128;
   }

   int sobel_orientation = 0;

   if (sobel_angle < 16 || sobel_angle >= (7 * 16)) {
         sobel_orientation = 2;
   } else if (sobel_angle >= 16 && sobel_angle < 16 * 3) {
         sobel_orientation = 1;
   } else if (sobel_angle >= 16 * 3 && sobel_angle < 16 * 5) {
         sobel_orientation = 0;
   } else if (sobel_angle > 16 * 5 && sobel_angle <= 16 * 7) {
         sobel_orientation = 3;
   }

   int16_t sobel_magnitude = input_magnitude[gid];
   /* Non-maximum suppression
      * Pick out the two neighbours that are perpendicular to the
      * current edge pixel */
   int16_t neighbour_max = 0;
   int16_t neighbour_max2 = 0;
   switch (sobel_orientation) {
         case 0:
            neighbour_max =
               input_magnitude[idx(x, y, width, height, 0, -1)];
            neighbour_max2 =
               input_magnitude[idx(x, y, width, height, 0, 1)];
            break;
         case 1:
            neighbour_max =
               input_magnitude[idx(x, y, width, height, -1, -1)];
            neighbour_max2 =
               input_magnitude[idx(x, y, width, height, 1, 1)];
            break;
         case 2:
            neighbour_max =
               input_magnitude[idx(x, y, width, height, -1, 0)];
            neighbour_max2 =
               input_magnitude[idx(x, y, width, height, 1, 0)];
            break;
         case 3:
         default:
            neighbour_max =
               input_magnitude[idx(x, y, width, height, 1, -1)];
            neighbour_max2 =
               input_magnitude[idx(x, y, width, height, -1, 1)];
            break;
   }
   // Suppress the pixel here
   if ((sobel_magnitude < neighbour_max) ||
         (sobel_magnitude < neighbour_max2)) {
         sobel_magnitude = 0;
   }

   /* Double thresholding */
   // Marks YES pixels with 255, NO pixels with 0 and MAYBE pixels
   // with 127
   uint8_t t = 127;
   if (sobel_magnitude > threshold_upper) t = 255;
   if (sobel_magnitude <= threshold_lower) t = 0;
   output[gid] = t;

}