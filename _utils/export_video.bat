call C:/Users/aless/anaconda3/Scripts/activate.bat C:/Users/aless/anaconda3
call activate stylegan3

set path=00024-stylegan2-incisioni-h3_m-1024x1024-gpus1-batch32-gamma6.6
set model_name=network-snapshot-000312-Gs.pkl
set test_name=double-symm-90
set test_prefix=illustrazioni-h2

@REM cd ..


@REM python ada_src/_genSGAN2.py --model "models\%path%\%model_name%" --out_dir "_out\out_SGAN2\%path%\%test_name%" --frames "90-90" --size "1080-3640" --scale_type 'symm' -n "1-2" --splitfine 0. --cubic --seed 0 --verbose

ffmpeg -framerate 30 -y -v warning -thread_queue_size 512 -i "_out\out_SGAN2\%path%\%test_name%\%06d.jpg" "_out\out_SGAN2\%test_prefix%_%test_name%.mp4"

conda list pandas
pause