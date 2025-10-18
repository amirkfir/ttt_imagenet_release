import argparse
from email.quoprimime import header_decode
import torch
from transformers.testing_utils import torch_device

from utils.misc import *
from utils.test_helpers import *
from utils.train_helpers import *
import torch.ao.quantization as quant


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='../imagenet-100/')
parser.add_argument('--shared', default="layer3")
########################################################################
parser.add_argument('--depth', default=-1, type=int) #18
parser.add_argument('--group_norm', default=32, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('--workers', default=8, type=int)
########################################################################
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--lr', default=0.1, type=float)
########################################################################
parser.add_argument('--resume', default = None)#default="results/resnet18_layer3_gn_100_classes")
parser.add_argument('--outf', default='.')
parser.add_argument('--rotation', default=True,type=bool)
parser.add_argument('--data_parallel', default=False,type=bool)




args = parser.parse_args()
my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
net, ext, head, ssh = build_model(args)

if args.resume is not None:
	print('Resuming from checkpoint..')
	ckpt = torch.load('%s/ckpt.pth' %(args.resume))


	# Create a new dict without 'module.' prefix
	net_dict = {}
	for k, v in ckpt["net"].items():
		name = k.replace("module.", "")  # remove `module.` prefix
		net_dict[name] = v

	head_dict = {}
	for k, v in ckpt["head"].items():
		name = k.replace("module.", "")  # remove `module.` prefix
		head_dict[name] = v

	net.load_state_dict(net_dict)
	head.load_state_dict(head_dict)


class CombinedModel(nn.Module):
	def __init__(self, net,ssh):
		super(CombinedModel, self).__init__()
		self.net = net
		# self.ssh = ssh

	def forward(self, x):
		return self.net(x)#,self.ssh(x)


device = torch.device("cuda")
_, trloader = prepare_train_data(args)
print("dataloader ready")
dl = next(iter(trloader))
image = dl[0].cuda()
# model = CombinedModel(net,ssh).to(device)
model = net.to(device)
quantization = False
if quantization:

	# 2️⃣ Fuse modules (Conv+ReLU, etc.)
	# quant.fuse_modules(model.net, [['conv1','bn1', 'relu']],inplace=True)
	# quant.fuse_modules(model.net.layer1, [['conv1','bn1', 'relu'],['conv2','bn2']],inplace=True)

	# 3️⃣ Specify quantization configuration
	model.qconfig = quant.get_default_qat_qconfig("fbgemm")

	# 4️⃣ Prepare model for QAT
	model_qat = quant.prepare_qat(model)

	# 5️⃣ Train as usual
	# train(model_qat)  # your training loop

	# 6️⃣ Convert to quantized model
	# # model_int8 = quant.convert(model_qat.cpu())
	# for name, module in model_int8.named_modules():
	# 	if isinstance(module, torch.nn.Conv2d) and module.bias is None:
	# 		with torch.no_grad():
	# 			module.bias = torch.nn.Parameter(torch.zeros(module.out_channels, dtype=module.weight.dtype))

torch.onnx.export(model,image,"model.onnx",opset_version=15)