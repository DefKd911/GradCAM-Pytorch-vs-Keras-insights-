import torch
import torch.nn.functional as F
from torchvision import models,transforms
from PIL import Image

class PytorchGradCAM:
    def __init__(self):
        self.model=models.resnet50(pretrained=True)
        self.model.eval()
        self.target_layer=self.model.layer4[2].conv3

        self.transform =transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ]
        )

    def generate_heatmap(self,img_path,target_class=None):

        img = Image.open(img_path).convert('RGB')
        input_tensor=self.transform(img).unsqueeze(0)
        
        gradients=[]
        activations=[]

        def backward_hook(module,grad_input,grad_output):
            gradients.append(grad_output[0])
        
        def forward_hook(module,input,output):
            activations.append(output)
        
        handle_backward=self.target_layer.register_backward_hook(backward_hook)
        handle_forward=self.target_layer.register_forward_hook(forward_hook)

        output=self.model(input_tensor)

        if target_class is None:
            target_class= output.argmax(dim=1).item()
        
        # Backward pass
        output[0,target_class].backward()

        # Generate heatmap
        grads = gradients[0].detach()
        acts = activations[0].detach()
        # Remove hooks
        handle_backward.remove()
        handle_forward.remove()

        pooled_grads = torch.mean(grads, dim=[0, 2, 3])

        for i in range(acts.shape[1]):
            acts[:,i,:,:]*=pooled_grads[i]
        
        heatmap=torch.sum(acts,dim=1).squeeze()
        heatmap=F.relu(heatmap)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = heatmap.cpu().numpy()

        return heatmap,img
    

       






