# Data Generation with Imagen

This repository contains simple data generation examples using [Imagen](https://github.com/lucidrains/imagen-pytorch) model. The experiments are conducted to generate real-life license plate and outdoor banners.

### Train Imagen
Train Imagen model using a specific dataset and save the trained model for later use during inference.
```python
python main.py --data = "lp" --pretrained = "path/to/checkpoint" --batch_size = 16 --im_size = 224
```

### Inference with Imagen
Generate images using the trained model by running the following script.
```python
python inference.py
```

### Real-life license plate generation

![image](https://user-images.githubusercontent.com/50166164/231911870-5bfec7cd-1d71-4acc-8070-14e14ceb355f.png)
![image](https://user-images.githubusercontent.com/50166164/231911898-59b26f34-8bdc-44f0-be97-24219a4742ab.png)
![image](https://user-images.githubusercontent.com/50166164/231911955-49f6bf9d-80da-4727-ab23-3b5ca7a73aa1.png)

![image](https://user-images.githubusercontent.com/50166164/231911972-9133aa79-90c2-4e57-a365-eed727a0cc92.png)
![image](https://user-images.githubusercontent.com/50166164/231912003-5a8dc503-ab53-4434-8328-00f0139de718.png)
![image](https://user-images.githubusercontent.com/50166164/231912062-1f7e8108-5f1e-49bb-864d-8e8e0fb34569.png)

### Outdoor banner generation

![image](https://user-images.githubusercontent.com/50166164/231912260-87c37e8a-3565-4379-975d-dc52a612a43f.png)
![image](https://user-images.githubusercontent.com/50166164/231912272-181add39-e1f6-45a5-adfb-742d6a71bcc4.png)
![image](https://user-images.githubusercontent.com/50166164/231912337-01b15981-8e07-4ae8-b56c-bf78e5dcec07.png)

![image](https://user-images.githubusercontent.com/50166164/231912359-a1337b6e-6c9c-4a7f-ba31-eb57dc2b8815.png)
![image](https://user-images.githubusercontent.com/50166164/231912416-d90bb809-3e37-4315-b2d9-1d7f32f756b6.png)
![image](https://user-images.githubusercontent.com/50166164/231912442-785a225d-fd0f-49ef-bb1d-9315f24ed6c8.png)

