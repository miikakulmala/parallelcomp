typedef uchar uint8_t;
typedef signed short int16_t;

__kernel void sobel(__global uint8_t* input,
                    __global int16_t* output_x,
                    __global int16_t* output_y,
                    int width, int height) {

   const int x = get_global_id(0); 
   const int y = get_global_id(1);

   output_x[y*width + x] = 255;
   output_y[y*width + x] = 255;

}

__kernel void PnM(__global int16_t* input_x,
                    __global int16_t* input_y,
                    __global uint8_t* output_phase,
                    __global int16_t* output_magnitude,
                    int width, int height) {

   const int x = get_global_id(0); 
   const int y = get_global_id(1);

   output_phase[y*width + x] = 255;
   output_magnitude[y*width + x] = input_x[y*width + x];

}

__kernel void nonMax(__global uint8_t* input_phase,
                    __global int16_t* input_magnitude,
                    __global uint8_t* output,
                    int width, int height,
                    int threshold_lower,
                    int threshold_upper) {

   int x = get_global_id(0);
   int y = get_global_id(1);

   output[y*width + x] = input_magnitude[y*width + x];

}
