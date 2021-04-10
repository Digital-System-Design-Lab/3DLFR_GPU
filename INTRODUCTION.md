# Introduction: Real-time Light field Rendering for Free Viewpoint Exploration

- This system is intended to provide an experience that feels like freely roaming a virtual space in real time. Free-viewpoint synthesis based on the light field rendering theory is accelerated by the GPU.

+ We acquire rays of 3-dimensional spaces by capturing dense 360-degree images. A GoPro fusion camera and a camera dolly from Edelkrone move and capture the light along the white path as shown below.

![image-20210411045908998](C:\Users\Lee_Seungho\AppData\Roaming\Typora\typora-user-images\image-20210411045908998.png)

+ We captures all the light which come through the 2D plane. To construct the “virtual view” at the arbitrary viewpoint, the lights required at the viewpoint are chosen and combined. 

![image-20210411050137701](C:\Users\Lee_Seungho\AppData\Roaming\Typora\typora-user-images\image-20210411050137701.png)

+ The square which is called `Light Field Unit(LFU)` is composed of four sides. Each side corresponds to the 2D plane that captures lights coming through. Synthesizing the view at the viewpoint inside the square is simplified to the process of selecting a light.

![image-20210411050504445](C:\Users\Lee_Seungho\AppData\Roaming\Typora\typora-user-images\image-20210411050504445.png)

+ Square-shaped `LFU` can be applied to any places you want to capture. No limit to expansion!

# System Summary

- ## Slice: Data transfer unit

  ***Slice*** is a group of pixel columns, is used as a unit of data movement and storage. The slice is made by dividing each image vertically as shown below.

  The advantage of transmitting in slice units is that transmission overhead can be prevented by transmitting the data portion for view synthesis.

<img src="C:\Users\Lee_Seungho\AppData\Roaming\Typora\typora-user-images\image-20210411055641370.png" alt="image-20210411055641370" style="zoom:50%;" />

+ ## Datapath

  (1) When the user’s current viewpoint is entered into the system, the ***slice manager*** calculates the range and amount of LF data required to synthesis the corresponding view and checks if there is any data already loaded into the ***device memory***.

  (2) Data is transferred from ***host memory*** to ***device memory***.

  (3) Using the LF data in the ***device memory***, a GPU performs the view synthesis. In the proposed system, view synthesis is the process of creating a novel view by selecting and combining pixels from slices stored in ***device memory***.

  (4) It may not be possible to store all required LF data in ***host memory***. Therefore, while performing view synthesis, at the same time, LF data required by neighboring viewpoints is transferred from ***storage*** to ***host memory***.

![image-20210411045349232](C:\Users\Lee_Seungho\AppData\Roaming\Typora\typora-user-images\image-20210411045349232.png)



- ## Implementation of Slice Manager

  Each slice has a unique ID. Host memory has LF data, a list of recently used slices, and a hashmap to manage IDs and memory addresses of slices.  ***H<sub>hashmap</sub>*** contains the address of the host memory where slices are stored, whereas ***H-D<sub>hashmap</sub>*** contains the address of the device memory where slices transferred from the host memory. Device memory is allocated for **D<sub>slice</sub>**, ***D<sub>hashmap</sub>***, and ***D<sub>output</sub>***. Slices transferred from host memory are stored in ***D<sub>slice</sub>***. When the GPU kernel for view synthesis is running, it can quickly access the slices in ***D<sub>slice</sub>*** by looking up the address of the ***D<sub>hashmap</sub>***. The synthesized view is saved in ***D<sub>output</sub>***. In Fig. 4, when a pixel column belonging to slice ID=23 is needed at the current viewpoint, the processing steps are shown from (1) to (5). First, the ***H<sub>hashmap</sub>*** is searched for ID=23. If the address value is NULL, this slice is not in the slice list and must be added. After writing the host memory address of slice ID=23 in the ***H<sub>hashmap</sub>*** (1), it is appended to the slice list (2). Then, the slice is transferred to ***D<sub>slice</sub>*** of device memory (3), and the address of ***D<sub>slice</sub>*** is written to ***H-D<sub>hashmap</sub>*** (4).  The same process is repeated for all slices required at the current viewpoint. Finally, the two hashmaps are synchronized by copying the ***H-D<sub>hashmap</sub>*** to the ***D<sub>hashmap</sub>*** (5). Since the GPU kernel cannot know the address of the device memory where the data transmitted from the host, the process of (5) is necessary. 

  <img src="C:\Users\Lee_Seungho\AppData\Roaming\Typora\typora-user-images\image-20210411045404579.png" alt="image-20210411045404579" style="zoom: 50%;" />

- ## Hierarchical Structure between Storage and Host Memory

  ### 1) LFU Window

  - Left figure show the structure of ***LFU Window***. All LF data are stored in storage, and only the 3x3 stacked LFU centered on the current LFU, called ***LFU Window***, is loaded to the host memory on the background. The current and next LFU windows are illustrated in red and blue, respectively. The newly loaded 7 blue LF data is overwritten in the space of the red LF data that do not belong to the blue LFU window.

  - Right figure shows vector representing the shortest distances that can go to eight neighboring LFUs at the current viewpoint inside the LFU. The priority when reading the LF data is determined in the order of the smallest vector size. In order to adaptively cope with the change in priority according to the dynamic viewpoint movement, LF data is divided and loaded in units of image groups.

    ​																			<img src="C:\Users\Lee_Seungho\AppData\Roaming\Typora\typora-user-images\image-20210411045430796.png" alt="image-20210411045430796" style="zoom:50%;" /><img src="C:\Users\Lee_Seungho\AppData\Roaming\Typora\typora-user-images\image-20210411045452007.png" alt="image-20210411045452007" style="zoom:50%;" />

    

  ### 2) Progressive Background Update

  - progressive LF update is proposed for smooth data movement from storage to host memory. To do this, each view of LF data is divided and managed in an interlaced format as shown below. 

    <img src="C:\Users\Lee_Seungho\AppData\Roaming\Typora\typora-user-images\image-20210411045547580.png" alt="image-20210411045547580" style="zoom:50%;" />

  - The flow chart shows the progressive LF update using the interlaced LF format. When the current LFU in the center is changed by the viewpoint movement, the LFU window slides to determine what data needs to be read from the storage(1).This includes even (or odd) fields that can make views of the current LFU complete and odd (or even) fields for new neighboring LFUs. While GPU rendering is being performed, even fields of the current LFU are read first. (2) to quickly change the rendered view from the current viewpoint to high quality. If the LF data of the full view for the current LFU is not yet prepared, the half resolution view from the odd fields is synthesized. At the same time, even fields are read continuously (3). When the reading of the even fields for the current LFU is finished, the full resolution view is now synthesized and the odd field of the neighboring LFU is read in the background. (4). 

![image-20210411045555114](C:\Users\Lee_Seungho\AppData\Roaming\Typora\typora-user-images\image-20210411045555114.png)