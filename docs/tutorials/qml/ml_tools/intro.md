# Introduction to Qadence ML Tools

Welcome to the Qadence `ML Tools` documentation! This submodule is designed to streamline your machine learning workflows—especially for quantum machine learning—by providing a set of robust tools for training, monitoring, and optimizing your models.

## What You'll Find in This Documentation

- **Trainer Class**  
  Learn how to leverage the versatile `Trainer` class to manage your training loops, handle data loading, and integrate with experiment tracking tools like TensorBoard and MLflow. Detailed guides cover:

    - Setting up training on both GPUs and CPUs.
    - Configuring single-process, multi-processing, and distributed training setups.

- **Gradient Optimization Methods**  
  Explore both gradient-based and gradient-free optimization strategies. Find examples demonstrating how to switch between these modes and how to use context managers for mixed optimization.

- **Custom Loss Functions and Hooks**  
  Discover how to define custom loss functions tailored to your tasks and use hooks to insert custom behaviors at various stages of the training process.

- **Callbacks for Enhanced Training**  
  Utilize built-in and custom callbacks to log metrics, save checkpoints, adjust learning rates, and more. This section explains how to integrate callbacks seamlessly into your training workflow.

- **Experiment Tracking**  
  Understand how to configure experiment tracking with tools such as TensorBoard and MLflow to monitor your model’s progress and performance.

## Getting Started

To dive in, explore the detailed sections below:

  - [Qadence Trainer Guide](./trainer.md)
  - [Training Configuration](./data_and_config.md)
  - [Callbacks for Trainer](./callbacks.md)
  - [Accelerator for Distributed Training](./accelerator.md)
  - [Training on GPU with Trainer](./GPU.md)
  - [Training on CPU with Trainer](./CPU.md)
