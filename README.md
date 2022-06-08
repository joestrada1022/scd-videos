# Source Camera Device Identification from Videos

This repository consists of code to reproduce the results reported in our paper. Experiments were conducted on 
the [VISION](https://lesc.dinfo.unifi.it/it/node/203) and the [QUFVD](https://ieeexplore.ieee.org/document/9713852/) 
data sets. Access to the published [open access paper](https://link.springer.com/article/10.1007/s42979-022-01202-0).

## Dataset Preparation
### 1. Preparation of VISION data set

#### Dataset download
To prepare the data sets, we assume the VISION data set (only videos) is downloaded and available in the original 
structure:

```
VISION  
  │  
  │─── D01  
  │     │  
  │     └─── videos  
  │             │  
  │             │─── flat  
  │             │     │  
  │             │     └─── flat_video1.mp4  
  │             │     └─── flat_video2.mp4  
  │             │─── flatWA  
  │             │     │  
  │             │     └─── flatWA_video1.mp4  
  │             │     └─── flatWA_video2.mp4  
  │             │─── ...  
  │             └─── outdoorYT  
  │─── D02  
  │─── ...  
  └─── D35  
   
```

#### Frame extraction

_Prerequisites_: FFmpeg library (https://ffmpeg.org/)

Extract the frames from videos in the VISION data set

- Execute `dataset/vision/frame_extraction.py` and set params `--input_dir="<path to the vision data set>"` and
` --output_dir="<path to an directory to save the extracted frames>"`. Refer to the [script](https://github.com/bgswaroop/scd-videos/tree/main/dataset/vision/frame_extraction.py) 
  for additional command line arguments and details. 

The structure of the resulting VISION frames data set is as follows:

```
VISION FRAMES DATASET
  │
  │─── D01
  │     │
  │     │─── flat_video1
  │     │       │
  │     │       └─── flat_video1_frame1.png
  │     │       └─── flat_video1_frame2.png
  │     └─── flatWA_video1
  │     └─── flatYT_video1
  └─── D02
```

### 2. Preparation of the QUFVD data set
#### Dataset download

QUFVD data set has three components, only the I-frames data set (`IFrameForEvalution20Class`) is considered for our experiments, which has 
the following structure: 

```
IFrameForEvalution20Class  
  │  
  │─── FrameDatabaseTesting  
  │     │  
  │     └─── CameraModelName  
  │             │  
  │             │─── Device1  
  │             │     │  
  │             │     └─── video1_frame1.jpg  
  │             │     └─── video1_frame2.jpg
  │             │          ...
  │             │     └─── video2_frame1.jpg
  │             │     └─── video2_frame2.jpg 
  │             └─── Device2   
  │─── FrameDatabaseTesting   
  └─── FrameDatabaseTraining  
```

#### Frame extraction
The I-frames are already extracted and divided into three sets as shown above. We make use of this default split.

## Training

Script `constrained_net/train/train.py` can be executed to train the ConstraindNet. Bash files to train the ConvNet and
ConstrainedNet on e.g. HPC Peregrine can be found in `constrained_net/bash/frames/*`.

Use param `--model_name` to set the name of your model. To continue training from a saved model, specify the path to the
particular model by using `--model_path`.

## Predicting

Script `constrained_net/predict/predict_flow.py` can be executed to automatically generate predictions on frame and
video level. Param `--input_dir` should point to a directory consisting of models (.h5 file). In my case, I saved a
model after every epoch. The script generates frame and video predictions for each model available in the input
directory. If you only want specific models to be evaluated, use param `--models` to specifiy the filenames of the
models (separated by a comma).

Script `constrained_net/predict/predict_flow.py` involves many steps, which I will further explain.

To create frame and video predictions, we first create two csv-files for every model in the input directory:

1. In `constrained_net/predict/predict_frames.py`, frames are predicted by the ConstrainedNet and subsequently saved to
   a csv-file.
2. In `constrained_net/predict/predict_videos.py`, the frame prediction csv-file (produced in step 1.) is loaded and
   used to predict videos by the majority vote. Video predictions are saved to a separate csv-file.

This eventually results in `K` frame prediction csv-files and `K` video prediction csv-files where `K` represents the
number of models in the input directory. To visualize the prediction results, we have to calculate the statistics for
every frame and video csv-file, which is done as follows:

3. In `constrained_net/predict/frame_prediction_statistics.py`, frame statistics (averages per scenarios, platforms,
   etc.) are generated for every frame prediction csv-file.
4. In `constrained_net/predict/video_prediction_statistics.py`, video statistics (averages per scenarios, platforms,
   etc.) are generated for every frame prediction csv-file.

This results in 1 frame statistics csv-file and 1 video statistics csv-file. The statistics files have as many rows as
there are models available in the input directory. Lastly, these files are used to visualize the results in:

5. `constrained_net/predict/frame_prediction_visualization.py` to visualize frame prediction results.
6. `constrained_net/predict/video_prediction_visualization.py` to visualize video prediction results.

Bash files to evaluate the ConstrainedNet on e.g. HPC Peregrine can be found in `constrained_net/bash/frames`.

### Citation   
```
@article{bennabhaktula2022source,
  title={Source Camera Device Identification from Videos},
  author={Bennabhaktula, Guru Swaroop and Timmerman, Derrick and Alegre, Enrique and Azzopardi, George},
  journal={SN Computer Science},
  volume={3},
  number={4},
  pages={1--15},
  year={2022},
  publisher={Springer}
}
```   