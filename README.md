# Video-Enhancement
This repository will serve as a foundation for applying digital image processing concepts, especially in videos, for a project at the Port of Pecém in Ceará.

- ```real_time_histogram.py``` displays the original video and the video after the application of some image processing method every certain number of frames. At the same time, dynamic image histograms are generated;
- ```preprocessing.py``` intended to contain digital image processing methods, primarily for enhancement purposes. Credits to [Luis Carlos](https://github.com/luiscarlo5);
- ```trackship.py``` provides a primary solution for calculating the speed of a ship based on the simulation video **part3.mp4**. The method analyzes the color difference between pixels from the platform to the ship and includes a series of methods to obtain the average and critical position of the ship. Credits to [
Carlos Victor](https://github.com/DlanorKnox);
- ```classic_segmentation.py``` serves the same purpose as the previous script, employing the principle of initially segmenting the ship. It also contains functions to simultaneously display various videos representing the color channel of the original video, plot the RGB and HSV color channels of a frame, and analyze a pair of frames. Credits to [Ermeson Alves](https://github.com/ermeson-alves).