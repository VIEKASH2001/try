"""
 * @author Gautam R Gare
 * @email gautam.r.gare@gmail.com
 * @create date 2021-12-03 22:09:24
 * @modify date 2021-12-03 23:06:14
 * @desc [description]
 * @referenced from: https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling
 """
 
 

import torch
import matplotlib.pyplot as plt

model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer2 = torch.optim.SGD(model.parameters(), lr=0.1)

cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=10, 
                T_mult=2, #1, 
                eta_min=0.001, 
                last_epoch=-1
            )
            
plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max', 
                factor = 0.25, #0.75, #0.25, #0.5 
                patience = 5, #patience=2, #patience=5, #patience=10,
                verbose=True, threshold=1e-3, threshold_mode='rel',
                cooldown=0, min_lr=1e-7, eps=1e-8
            )

cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, 
                base_lr=0.001, 
                max_lr=0.1, 
                step_size_up=2, #5,
                mode="exp_range",
                gamma=0.85
            )

triangular_scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, 
                base_lr = 1e-7, #0.001, 
                max_lr = 0.0001, #0.1, 
                step_size_up=10,
                mode="triangular2"
            )

step_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer2, #optimizer, 
                step_size=20, #10, #2, 
                gamma= 0.5 #0.1
            )

lrs = []
acc = 0

for i in range(100):
    
    # cosine_scheduler.step()
    # cyclic_scheduler.step()
    # triangular_scheduler.step()
    
    step_scheduler.step()
    print(step_scheduler.get_lr())
    # triangular_scheduler.max_lrs = step_scheduler.get_lr()
    # triangular_scheduler.step()

    cosine_scheduler.base_lrs = step_scheduler.get_lr()
    cosine_scheduler.eta_min = step_scheduler.get_lr()[0]/100
    cosine_scheduler.step()


    # acc+=1
    # if i%10 == 0:
    #     acc=acc/10
    # plateau_scheduler.step(acc)

    lrs.append(
        optimizer.param_groups[0]["lr"]
    )

plt.plot(lrs)
plt.show()



# import torch
# from torch.nn import Parameter
# from torch.optim import SGD

# model = [Parameter(torch.randn(2, 2, requires_grad=True))]
# optimizer = SGD(model, 0.1)

# scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# # scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100)
# scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.1, 1, step_size_up=100)

# values, values1, values2 = [], [], []
# for epoch in range(1000):
#     values.append(optimizer.param_groups[0]["lr"])
#     values1.append(scheduler1.get_last_lr()[0])
#     values2.append(scheduler2.get_last_lr()[0])
#     # optimizer.step()
#     # scheduler1.step(epoch)
#     scheduler1.step()
#     scheduler2.step()

# import matplotlib.pyplot as plt
# plt.plot(values, label='values')
# plt.plot(values1, label='values 1')
# plt.plot(values2, label='values 2')
# plt.legend()
# plt.show()