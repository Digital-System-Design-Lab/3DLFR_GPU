This is a demo version and may contain bugs. Light field data is required to run this program, but it is not included here. We will provide it ASAP.

This program was written in the MSVC v142 environment.

Dependencies : CUDA 11.2, GLEW, freeglut

-----

**3DLFR_GPU**

- **README.md** : This file

- **INTRODUCTION.md** : Introduction to what we did

- **data** 

  - **LFdata** (Not prepared yet)
    Light field data that we acquire while capturing real space. We captured it at the "BMW Driving Center" in Incheon, Korea, and thanks again for the rental of the place.
  - **pixel_selection**
    Information on pixels required at a location inside the LFU.  (Which image from which column to get the pixels) 
    You can choose between two resolutions: 4096x2048 and 7680x4320.
  - **Map configuration_BMW driving center_Incheon_Korea.xlsx** 
    Index information of the light field data constituting the virtual space of 600x5600 cm<sup>2</sup>. 

- **sources**
  
  source code of this. You can run the program just by setting the resolution in the following block of code, setting the path to the LF data and the path containing the pixel_selection file.
  
  ```c++
    #define RESOLUTION 4
    ...
    #define PATH_LF "S:/BMW_4K/"
    #define PATH_PIXEL_RANGE "S:/PixelRange_4K/"
  ```
  
  ***NOTE:*** The last argument among the constructors of the `LF_Renderer` class determines whether to use the LFU window mode or explore only inside a single LFU. 
  
  ```c++
    LF_Renderer renderer(PATH_LF, PATH_PIXEL_RANGE, WIDTH, HEIGHT, LF_length, num_LFs, dpp, stride, curPosX, curPosY, true);
  ```

- **lib/bin**

  libs and dlls for glew and freeglut.

