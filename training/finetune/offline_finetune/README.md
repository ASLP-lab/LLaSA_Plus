 

This is a sample code demonstrating how to perform offline fine-tuning by adding MTP Modules. You can refer to the `data_instruction` folder for dataset guidance.

### Full-Parameter Training

To fine-tune using all parameters, run the following command:

```bash
torchrun --nproc_per_node=4 finetune_offline_mtp2_notcat.py
```

> **Note:** If you are using your own dataset, you may need to adjust hyperparameters such as the learning rate, configuration, etc., to achieve the best results.

