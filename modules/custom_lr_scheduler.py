import numpy as np

def custom_lr_schedule_wrapper(num_epochs, 
                               lr_start, 
                               warmup_step, 
                               warmup_lr_end, 
                               annealing_step, 
                               annealing_lr_end, 
                               num_cicles, 
                               max_cicles_lr):
    """
    Wrapper function to create a custom learning rate scheduler.
    The learning rate schedule is composed of a warmup phase, an annealing phase and a cyclic phase.
    The warmup phase is a linear increase from lr_start to warmup_lr_end in warmup_step epochs.
    The annealing phase is a cosine annealing from warmup_lr_end to annealing_lr_end in annealing_step epochs.
    The cyclic phase is a cosine annealing from annealing_lr_end to max_cicles_lr in num_cicles cycles.

    Parameters
    ----------
    num_epochs : int
        Total number of epochs.
    lr_start : float
        Initial learning rate.
    warmup_step : int
        Number of epochs for the warmup phase.
    warmup_lr_end : float
        Final learning rate of the warmup phase.
    annealing_step : int
        Number of epochs for the annealing phase.
    annealing_lr_end : float
        Final learning rate at the end of the annealing phase.
    num_cicles : int
        Number of cycles in the cyclic phase.
    max_cicles_lr : float
        Maximum learning rate in the cyclic phase.

    Returns
    -------
    custom_lr_schedule : function
        Custom learning rate scheduler.
    """
    
    def custom_lr_schedule(epoch):
        if epoch < warmup_step:
            return lr_start + epoch / warmup_step * (warmup_lr_end - lr_start)
        elif epoch < warmup_step + annealing_step:
            return annealing_lr_end + 0.5 * (warmup_lr_end - annealing_lr_end) * (1 + np.cos((epoch - warmup_step) / annealing_step * np.pi))
        else:
            cycle_length = (num_epochs - warmup_step - annealing_step) // num_cicles
            delta_y = annealing_lr_end - max_cicles_lr
            delta_x = cycle_length - 1
            slope = delta_y / delta_x
            return max_cicles_lr + slope * ((epoch - warmup_step - annealing_step) % cycle_length)

    
    return custom_lr_schedule

