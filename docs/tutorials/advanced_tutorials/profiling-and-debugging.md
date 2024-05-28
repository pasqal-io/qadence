

For this to work, you'll have to have the right to access perf counters on your machine (for example with sudo or inside a Docker container).
See: https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/index.html.

```bash
$ pip install nvidia-pyindex
$ pip install nvidia-dlprof[pytorch]
```

Make sure that your entrypoint is an executable script.
That means that it must start with a she-bang on the top line, e.g.
```python
#!/bin/python
```
and have execution rights
```bash
$ chmod +x your_script.py
```

Lastly it's recommended to add
```python
import nvidia_dlprof_pytorch_nvtx
nvidia_dlprof_pytorch_nvtx.init()
```
To your script in the beginning enable extra annotations of PyTorch functions.

You can then use dlprof to profile.
```
dlprof --mode=pytorch your_script.py
```

```
PYQ_LOG_LEVEL=info QADENCE_LOG_LEVEL=debug dlprof --mode=pytorch examples/backends/differentiable_backend.py
```

For example to achieve this through Docker we can start a session in the shell, also mounting our local Qadence version and PyQTorch (Both optional, but you probably need to mount your script at least)
```bash
$ docker run --rm --gpus=1 --shm-size=1g --ulimit memlock=-1 \
 --ulimit stack=67108864 -it -p8000:8000 -v./:/opt/qadence -v ../PyQ:/opt/pyqtorch pytorch:24.02-py3 bash
```
(You may need to jump through extra hoops to make Docker access the GPUs if you have error messages like
`docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].`)

After this you should have a shell inside the container and you can
```
root@2a85826c4e7b:/workspace# cd /opt/qadence/
root@2a85826c4e7b:/opt/qadence# pip3 install -e .
root@2a85826c4e7b:/opt/qadence# pip3 install -e ../pyqtorch/
root@2a85826c4e7b:/opt/qadence# pip3 install nvidia-dlprof[pytorch]
root@2a85826c4e7b:/opt/qadence# PYQ_LOG_LEVEL=debug QADENCE_LOG_LEVEL=debug dlprof --mode=pytorch --nsys_opts="-t cuda,nvtx,cublas,cusparse,cusparse-verbose,cublas-verbose --force-overwrite true" examples/models/quantum_model.py
```

Where we have `--force-overwrite true` to always store the latest profiling result (hence you must rename the file) if you wish to keep several. We add `cublas,cusparse,cusparse-verbose,cublas-verbose` do get more details
about the numerical backend pacakges being used.
