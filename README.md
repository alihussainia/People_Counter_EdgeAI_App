# PeopleCounter

The "Deploy a People Counter App at the Edge" is a project of the Udacity "Intel® Edge AI for IoT Developers Nanodegree" Program which serves as an app for counting people using computer vision based Edge AI Computing.

The input is an video file, on which inference will be made i.e. the app will create a boundary box around the detected persons, count the different people as they are detected, keep track of the average presence time and publish these stats to a MQTT server.

## Prerequisites

- [Intel OpenVino toolKit](https://software.intel.com/en-us/openvino-toolkit) 
- [OpenCV](https://opencv.org)
- [FFMPEG server](https://trac.ffmpeg.org/wiki/ffserver)
- [Mosca server](https://github.com/moscajs/mosca)
- [GUI webservice]

## Usage
1. Open four Terminals.

2. Get the MQTT broker installed and running.
     ```
     cd webservice/server/node-server
     node ./server.js
     ```
     You should see a message that Mosca server started.


3. Get the UI Node Server running.

     ```
     cd webservice/ui
     npm run dev
     ```
     After a few seconds, you should see webpack: Compiled successfully.
     


4. Start the ffserver
     ```
     sudo ffserver -f ./ffmpeg/server.conf
     ```

5. Running the App
    - If using Udacity's Workspace, don't forget to click the SOURCE button.
    - shortcuts:
      ```
      export input_video = resources/Pedestrian_Detect_2_1_1.mp4
      export input_img = resources/snap1.png
      export model_lite = models/ssdlite_mobilenet_v2_coco.xml
      export cpu_x = /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extenson_sse4.so
      ```
      
      
     * For Video File:
   
`python main.py -i input_video -m model_lite -l cpu_x -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm`

     * For image file : 

`python main.py -i input_img -m model_lite -l cpu_x -d CPU -pt 0.6`
## Output
- A simple image file named `single_image.png`. For example:
![](resources/single_image.png?raw=true)

## AI Models used

The models have been added in the models directory. 

- [ssd_mobilenet_v2_coco.xml](models/ssd_mobilenet_v2_coco.xml)
- [ssdlite_mobilenet_v2_coco.xml](models/ssdlite_mobilenet_v2_coco.xml)


## Learning

The final drawn conclusion after testing the scripts with various models is:

- It is best suitable with an OpenVino intermediate representation (IR). 
- It also supports Tensorflow frozen models, but with poor results in terms of accuracy and inference time.