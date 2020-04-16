python main.py housedet --exp_id ct_det_house_res_grass --arch res_101 --batch_size 16 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 4

# test
python test.py housedet --exp_id ct_det_house_res_grass --arch res_101 --keep_res --resume