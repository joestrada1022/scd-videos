REM Changing directory to the target directory
cd D://Datasets/vision/
REM starting rsync
bash -c "rsync -avzhr -P --stats 'p288722@pg-gpu.hpc.rug.nl:/scratch/p288722/datasets/vision/all_I_frames' ."
