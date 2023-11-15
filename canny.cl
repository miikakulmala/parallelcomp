__kernel void sobel(__global int8* input,
                    __global int16* output_x,
                    __global int16* output_y,
                    int width, int height) {

   const int x = get_global_id(0); 
   const int y = get_global_id(1);

   output_x[y*width + x] = 255;
   output_y[y*width + x] = 255;

}

__kernel void PnM(__global int16* input_x,
                    __global int16* input_y,
                    __global int8* output_phase,
                    __global int16* output_magnitude,
                    int width, int height) {

   const int x = get_global_id(0); 
   const int y = get_global_id(1);

   output_phase[y*width + x] = 255;
   output_magnitude[y*width + x] = 255;

}

__kernel void nonMax(__global int8* input_phase,
                    __global int16* input_magnitude,
                    __global int8* output,
                    int width, int height,
                    int threshold_lower,
                    int threshold_upper) {

   int x = get_global_id(0);
   int y = get_global_id(1);

   output[y*width + x] = y;

}
