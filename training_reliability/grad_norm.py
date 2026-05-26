from training_reliability import Warning


def check_grad_norm(step, grad_norm, threshold=10.0):
    if grad_norm > threshold:
        return Warning(
            step=step,
            monitor='grad_norm',
            message='gradient spike',
            values={'grad_norm': grad_norm, 'threshold': threshold},
        )
    if grad_norm < 1e-6:
        return Warning(
            step=step,
            monitor='grad_norm',
            message='vanishing gradients',
            values={'grad_norm': grad_norm},
        )
    return None
