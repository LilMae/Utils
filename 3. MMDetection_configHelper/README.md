# Config Manual

## **1.1 Config 확인하기**

만들어진 config에 대한 자세한 설정은 

```python
python tools/misc/print_config.py config파일
```

을 통해 확인할 수 있다.

---

## 1.2 Config 파일 구조

`config/_base_` 파일은 크게 4가지 기본구조를 가지고 있다.

1. dataset
2. model
3. schedulr
4. default_runtime

`_base_` 를 통해 만들어져 있는 모델을 이식받아 사용할 수 있게 된다.

이식받아 사용하는 모델을 `primitive` config라고 하는데, 한번에 하나의 primitive config만 사용할 수 있다.

---

## 1.3 모델을 제외한 수정사항들

### 1) Basic 설정

→ 기존의 모델을 사용하여 작업을 수행할 때, Customdataset에 맞춰주기 위해 필요한 설정들

따라서 맨위에 기존 모델을 입력해주어야 한다.

```python
_base_ = #모델 설정이 위치한 파일경로
```

`_base_` (str)  : 사용하고자 하는 모델의 파일경로

ex)_*base_* = './configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py'

---

**데이터셋 정보 입력**

> Custom Dataset을 사용하기 위해 다음과 같은 정보가 필요함

```python
dataset_type = #데이터셋 포멧
data_root = #데이터셋이 위치한 폴더 경로
classes = #데이터셋에 존재하는 label정보
```

`dataset_tpye` (str) : 데이터셋 포멧을 명시

'COCODataset' , 'VOCDataset' 

`data_root` (str) : 데이터셋을 담고 있는 폴더 경로

ex)  data_root = './data_dataset_converted/'

`classes` (tuple) : 데이터셋에 존재하는 레이블들

ex) classes = ('microcystis', 'synedra', 'staurastrum', 'pediastrum', 'oscillatoria', 'anabaena', 'aphanizomenon')

---

**모델 일부 수정**

> 검출하고자하는 label개수와 맞추어 모델 전반적인 부분을 수정해야하지만, MMDetction에서는 roi_head내의 일부분만 바꾸는 것으로 모델 전반을 자동으로 조정해준다.

```python
model = dict(
	roi_head=dict(
		bbox_head = dict(num_classes=len(classes))
		mask_head = dict(num_classes=len(classes))
	)	
)
```

`model.roi_head 내에서 수정할 부분` 

`bbox_head` : num_classes 에 해당하는 값으로 len(classes)를 넣어준다.

`mask_head` : num_classes 에 해당하는 값으로 len(classes)를 넣어준다.

---

**데이터 로드** 

> 데이터는 train, val, test로 나누어 구성되어 있어야 하며, 각각의 데이터에 대한 정보를 입력해주어야 한다.

```python
data = dict(
    train=dict(
        img_prefix= #데이터 경로
        classes=classes,
        ann_file= #annotation 정보를 담고 있는 json파일의 경로
		),
    val=dict(
        img_prefix= #데이터 경로
        classes=classes,
        ann_file= #annotation 정보를 담고 있는 json파일의 경로
		),
    test=dict(
        img_prefix= #데이터 경로
        classes=classes,
        ann_file= #annotation 정보를 담고 있는 json파일의 경로
		)
	)
```

`img_prefix`  (str) : 데이터가 위치한 폴더(COCO의 경우 JEPGImages 폴더)의 **상위폴더 의 경로**

**위에서 만들어 놓은 data_root를 사용해서 쉽게 접근한다.**

ex) img_prefix = data_root +'train/'

`classes` (tuple) : 데이터에 포함된 label정보를 입력

**위에서 만들어 놓은 classes**를 사용하여 쉽게 접근

ex) classes = classes

`ann_file` (str) : annotation 정보를 담고있는 json 파일의 경로

 **위에서 만들어 놓은 data_root를 사용해서 쉽게 접근한다.**

ex) ann_file = data_root + 'train/annotations.json'

---

**저장 및 불러오기**

> 기본적인 모델 저장 및 불러오기에 사용되는

```python
work_dir = #학습 log, 학습된 모델이 저장될 폴더
load_from = #학습된 모델을 불러와서 사용하는 경우에 사용
runner = #학습횟수
```

`work_dir` (str) : 학습 log 및 학습된 모델이 저장될 폴더의 경로

ex) work_dir = './workspace'

`load_from` (str) : 학습된 모델을 불러오는 경우, 저장된 가중치의 경로

ex) load_from = './workspace/latest.pth'

`runner` (dict) : 최대 학습 횟수

ex) runner = dict(max_epochs=24)

### 2) Optional 설정

---

**데이터 전처리 부분 - 정규화**

```python
img_norm_cfg =  #이미지 정규화에 필요한 변수들
pipeline= [
	...
	img_norm_cfg,
	...
]
```

`img_norm_cfg` (dict) : 이미지 정규화에 필요한 파라미터들

ex)

img_norm_cgf = dict(type='Normalize',

mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

`pipeline` : 위와 같이 파라미터 수정을 데이터 전처리에 반영하기 위해 pipeline에 넣어준다.

---

**학습 관련 부분** 1) **Optimizer**

```python
optimizer = dict(
	type = #사용하고자 하는 Optimizer
	... #해당 Optimizer에서 사용하는 parameter들
)
```

`optimizer` (dict) : 파이토치에서 사용하는 Optim 중 하나를 선정하고

[https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim)

ex)

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

---

**학습 관련 부분 2) lr_config**

```python
lr_config = 
	dict(policy = #사용하고자 하는 스케줄러
			warmup = #가중치 warmup 방식
			warmup_iters = #warmup 지속 길이
			warmup_ratio = #warmup 사용시 초기 lr 설정
			warmup_by_epoch = #매 학습마다 warmup 사용 유무
			... #스케줄러별로 추가되는 파라미터들
	)
```

`lr_config` (dict) : 학습 스케줄러와 필요한 파라미터들을 전달

`policy` (str) : 사용하고자 하는 스케줄러

`warmup` (str) : 가중치 warmup 방식

'constant' , 'linear', 'exp' 중 하나를 선택하여 사용

`warmup_iter` (int) : warmup이 지속되는 epoch 길이

`warmup_ratio` (float) : wramup학습에서 시작 lr은 warmup_ratio * initial_lr

`warmup_by_epoch` (bool) : 매 학습마다 warmup을 수행 유무

`추가 파라미터들` (dict) : policy별로 필요한 파라미터들이 추가적으로 존재

**지원되는 policy와 파라미터**

- Step

    dict(policy='Step', step, gamma=0.1, min_lr=None)

    Args:

    step (int | list[int]): Step to decay the LR. If an int value is given,

    regard it as the decay interval. If a list is given, decay LR at

    these steps.

    gamma (float, optional): Decay LR ratio. Default: 0.1.

    min_lr (float, optional): Minimum LR value to keep. If LR after decay

    is lower than `min_lr`, it will be clipped to this value. If None

    is given, we don't perform lr clipping. Default: None.

- Poly

    dict(policy='Poly', power=1., min_lr=0.)

- Inv

    dict(policy='Inv', gamma, power=1.)

- CosineAnnealing

    dict(policy='CosineAnnealing', min_lr=None, min_lr_ratio=None)

- FlatCosineAnnealing

    dict(policy='FlatCosineAnnealing', start_percent=0.75, min_lr=None, min_lr_ratio=None)

    Modified from https://github.com/fastai/fastai/blob/master/fastai/callback/schedule.py#L128 # noqa: E501

    Args:

    start_percent (float): When to start annealing the learning rate

    after the percentage of the total training steps.

    The value should be in range [0, 1).

    Default: 0.75

    min_lr (float, optional): The minimum lr. Default: None.

    min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.

    Either `min_lr` or `min_lr_ratio` should be specified.

    Default: None.

- CosineRestart

    dict(policy='CosineRestart', periods, restart_weights=[1], min_lr=None, min_lr_ratio=None)

    Args:

    periods (list[int]): Periods for each cosine anneling cycle.

    restart_weights (list[float], optional): Restart weights at each

    restart iteration. Default: [1].

    min_lr (float, optional): The minimum lr. Default: None.

    min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.

    Either `min_lr` or `min_lr_ratio` should be specified.

    Default: None.

- Cyclic

    dict(policy='Cyclic', by_epoch=False, target_ratio=(10, 1e-4), cyclic_times=1, step_ratio_up=0.4, anneal_strategy='cos')

    Args:

    by_epoch (bool): Whether to update LR by epoch.

    target_ratio (tuple[float]): Relative ratio of the highest LR and the

    lowest LR to the initial LR.

    cyclic_times (int): Number of cycles during training

    step_ratio_up (float): The ratio of the increasing process of LR in

    the total cycle.

    anneal_strategy (str): {'cos', 'linear'}

    Specifies the annealing strategy: 'cos' for cosine annealing,

    'linear' for linear annealing. Default: 'cos'.

- OneCycle

    dict(policy='OneCycle', max_lr, total_steps=None, pct_start=0.3, anneal_strategy='cos', div_factor=25, final_div_factor=1e4, three_phase=False)

    Args:

    max_lr (float or list): Upper learning rate boundaries in the cycle

    for each parameter group.

    total_steps (int, optional): The total number of steps in the cycle.

    Note that if a value is not provided here, it will be the max_iter

    of runner. Default: None.

    pct_start (float): The percentage of the cycle (in number of steps)

    spent increasing the learning rate.

    Default: 0.3

    anneal_strategy (str): {'cos', 'linear'}

    Specifies the annealing strategy: 'cos' for cosine annealing,

    'linear' for linear annealing.

    Default: 'cos'

    div_factor (float): Determines the initial learning rate via

    initial_lr = max_lr/div_factor

    Default: 25

    final_div_factor (float): Determines the minimum learning rate via

    min_lr = initial_lr/final_div_factor

    Default: 1e4

    three_phase (bool): If three_phase is True, use a third phase of the

    schedule to annihilate the learning rate according to

    final_div_factor instead of modifying the second phase (the first

    two phases will be symmetrical about the step indicated by

    pct_start).

    Default: False

---

**학습관련 부분 3) workflow**

```python
workflow = [
		#phase와 epoch의 tuple로 구성된 원소들
	]
```

`workflow`  (dict) : (phase, epochs)의 튜플로 구성된 원소를 가지며, phase가 epochs만큼 진행된다.

ex)

workflow = [ ('train', 10), ('val' , 1)] 

→ 10번의 학습 epoch과 1번의 검증 epoch이 반복된다.

---

**Log 관련 부분**

```python
log_config = dict(
		interval = #Log를 남기는 간격
		hooks = #Log를 만드는 방법
	)
```

`log_config` (dict) : 학습 중 Log를 만드는 방법과 관련한 정보

`inverval` (int) : Log를 남기는 간격(epoch)을 저장

ex) interval = 50

`hooks` (list) : log를 남기는 방법을 설정, 몇가지 선택지가 제공되고 이에 맞는 파라미터들을 dict형태로 입력한다.

여러가지 로그를 한번에 사용할 수 있기 때문에, 여러개의 로그와 파라미터를 dict로 담은 list로 입력

ex) hooks=[dict(type='TextLoggerHook')]

**지원되는 LogHooker와 파라미터**

- WandbLoggerHook

    dict(type='WandbLoggerHook', init_kwargs=None, interval=10, ignore_last=True, reset_flag=False, commit=True, by_epoch=True, with_step=True)

- TextLoggerHook

    dict(type='TextLoggerHook', by_epoch=True, interval=10, ignore_last=True, reset_flag=False, interval_exp_name=1000)

- TensorboardLoggerHook

    dict(type='TensorboardLoggerHook', log_dir=None, interval=10, ignore_last=True, reset_flag=False, by_epoch=True)

- PaviLoggerHook

    dict(type='PaviLoggerHook',init_kwargs=None, add_graph=False, add_last_ckpt=False, interval=10, ignore_last=True, reset_flag=False, by_epoch=True, img_key='img_info')

- NeptuneLoggerHook

    dict(type='NeptuneLoggerHook', init_kwargs=None, interval=10, ignore_last=True, reset_flag=True, with_step=True, by_epoch=True)

- MlflowLoggerHook

    dict(type='MlflowLoggerHook', exp_name=None, tags=None, log_model=True, interval=10, ignore_last=True, reset_flag=False, by_epoch=True)

- DvcliveLoggerHook

    dict(type='DvcliveLoggerHook', path, interval=10, ignore_last=True, reset_flag=True, by_epoch=True)

---

# 방명록

유민: 대단해요! 1star드립니다. ★