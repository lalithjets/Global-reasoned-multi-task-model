import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


def demo_basic(rank, world_size):
    
    print(f"Running basic DDP example on rank {rank}.")

    '''
    setting master pc for sync all the process
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


    '''
    # declare the empty model, can also load the pre-trained model.
    then, pass the model to ddp, it helps train the model using parallel gpu
    and sync it back into one model.
    '''
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    '''
    declare loss function and optimizer
    '''
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    '''
    for trianing the model
    here, you can do the epoch loop for forward and backward propagation.
    '''
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()


    '''
    here, you can save epoch with a condition 
    "if gpu == 0": save checkpoint. 
    ddp helps keep master copy of model in all gpu, so, it doesn't matter which gpu is set in the if state.
    '''
    '''
    end of training
    '''
    dist.destroy_process_group()


def demo_checkpoint(rank, world_size):
    
    print(f"Running DDP checkpoint example on rank {rank}.")

    '''
    setting master pc for sync all the process
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    '''
    # declare the empty model, can also load the pre-trained model.
    then, pass the model to ddp, it helps train the model using parallel gpu
    and sync it back into one model.
    '''
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    '''
    declare loss function and optimizer
    '''
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    '''
    saving a checkpoint 
    "if gpu == 0": save checkpoint. 
    ddp helps keep master copy of model in all gpu, so, it doesn't matter which gpu is set in the if state.
    '''
    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    
    '''
    load checkpoint
    '''
    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=map_location))

    '''
    for trianing the model
    here, you can do the epoch loop for forward and backward propagation.
    '''
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    dist.destroy_process_group()



if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    
    mp.spawn(demo_basic,args=(8,),nprocs=8,join=True)
    mp.spawn(demo_checkpoint,args=(8,),nprocs=8,join=True)
      


