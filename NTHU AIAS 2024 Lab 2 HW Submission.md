# NTHU_111065541_張騰午 AIAS 2024 Lab 2 HW Submission


[toc]

## Gitlab code link

- https://course.playlab.tw/git/Tiger_Chang/lab02/-/tree/main/hw2 - 


## HW 2-1 Model Analysis Using Pytorch

### 2-1-1. Calculate the number of model parameters：

#### Code
```
total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters: ", total_params)
```


#### Execution Result
:::info
![](https://course.playlab.tw/md/uploads/205fa68a-d91b-44d7-94e6-e5944be89dd2.png)
:::


### 2-1-2. Calculate memory requirements for storing the model weights.
#### Code
```
param_size = sum(p.numel() * p.element_size() for p in model.parameters())
print("Total memory for parameters: ", param_size)
```

#### Execution Result
:::info
![](https://course.playlab.tw/md/uploads/31822d20-7608-4b7d-acb1-cac8e246bc53.png) 
:::


### 2-1-3. Use Torchinfo to print model architecture including the number of parameters and the output activation size of each layer 
#### Code
```
print(torchinfo.summary(model, (3, 224, 224), batch_dim=0, col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))
```

#### Execution Result
:::info
![](https://course.playlab.tw/md/uploads/02f7cbe3-aa57-4985-9589-5b8c4a43ddb9.png)
 ![](https://course.playlab.tw/md/uploads/3bc9571b-b8cd-45a2-8bda-238805ff8498.png)
![](https://course.playlab.tw/md/uploads/2929289b-722c-4c68-b076-600f28483c29.png)
![](https://course.playlab.tw/md/uploads/6ea9edcc-563d-4074-adfa-f71284bf6200.png)
![](https://course.playlab.tw/md/uploads/c8901e49-bfb9-45f3-b348-942d7d394d67.png)
![](https://course.playlab.tw/md/uploads/ca5a01de-cef0-4478-a926-db1aed99d3b1.png)

:::


### 2-1-4. Calculate computation requirements
#### Code
```
def calculate_output_shape(input_shape, layer):
    # Calculate the output shape for Conv2d, MaxPool2d, and Linear layers
    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
        kernel_size = (
            layer.kernel_size
            if isinstance(layer.kernel_size, tuple)
            else (layer.kernel_size, layer.kernel_size)
        )
        stride = (
            layer.stride
            if isinstance(layer.stride, tuple)
            else (layer.stride, layer.stride)
        )
        padding = (
            layer.padding
            if isinstance(layer.padding, tuple)
            else (layer.padding, layer.padding)
        )
        dilation = (
            layer.dilation
            if isinstance(layer.dilation, tuple)
            else (layer.dilation, layer.dilation)
        )

        output_height = (
            input_shape[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
        ) // stride[0] + 1
        output_width = (
            input_shape[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
        ) // stride[1] + 1
        return (
            layer.out_channels if hasattr(layer, "out_channels") else input_shape[0],
            output_height,
            output_width,
        )
    elif isinstance(layer, nn.Linear):
        # For Linear layers, the output shape is simply the layer's output features
        return (layer.out_features,)
    else:
        return input_shape


def calculate_macs(layer, input_shape, output_shape):
    # Calculate MACs for Conv2d and Linear layers
    if isinstance(layer, nn.Conv2d):
        kernel_ops = (
            layer.kernel_size[0]
            * layer.kernel_size[1]
            * (layer.in_channels / layer.groups)
        )
        output_elements = output_shape[1] * output_shape[2]
        macs = int(kernel_ops * output_elements * layer.out_channels)
        return macs
    elif isinstance(layer, nn.Linear):
        # For Linear layers, MACs are the product of input features and output features
        macs = int(layer.in_features * layer.out_features)
        return macs
    else:
        return 0

input_shape = (3, 224, 224)
total_macs = 0
for name, layer in model.named_modules():
    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.ReLU, nn.Linear)):
        output_shape = calculate_output_shape(input_shape, layer)
        macs = calculate_macs(layer, input_shape, output_shape)
        total_macs += macs
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            print(
                f"Layer: {name}, Type: {type(layer).__name__}, Input Shape: {input_shape}, Output Shape: {output_shape}, MACs: {macs}"
            )
        elif isinstance(layer, nn.MaxPool2d):
            # Also print shape transformation for MaxPool2d layers (no MACs calculated)
            print(
                f"Layer: {name}, Type: {type(layer).__name__}, Input Shape: {input_shape}, Output Shape: {output_shape}, MACs: N/A"
            )
        input_shape = output_shape  # Update the input shape for the next layer

print(f"Total MACs: {total_macs}")
```

#### Execution Result
:::info
![](https://course.playlab.tw/md/uploads/b1955779-3764-42f0-927e-b4ca4214e973.png) 
:::

### 2-1-5. Use forward hooks to extract the output activations of  the Conv2d layers.
#### Code
```
# Define a hook function
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


# Dictionary to store activations from each layer
activation = {}

for layer_name, layer in model.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        # Register forward hook
        layer.register_forward_hook(get_activation(layer_name))

# Run model inference
data = torch.randn(1, 3, 224, 224)
output = model(data)

# Access the saved activations
for layer in activation:
    print(f"Activation from layer {layer}: {activation[layer].shape}")
```

#### Execution Result
:::info
![](https://course.playlab.tw/md/uploads/1b6dfef2-7a3e-4b37-bc99-ad7525b6c6f7.png) 
:::

## HW 2-2 Add more statistics to analyze the an ONNX model Using sclblonnx

### 2-2-1. model characteristics
#### Code
```
def get_attribute(graph: xpb2.GraphProto):
    print("2-2-1. model characteristics\n")
    try:
        conv_attr = []
        for i, node in enumerate(inferred_model.graph.node):
                if node.name == "":
                    inferred_model.graph.node[i].name = str(i)
        # get the idx_list
        # idx_list = get_op_type_name_and_idx(graph)
        node_nlist = [k.name for k in graph.node]
        idx_list = {}
        for node in graph.node:
            if node.op_type in idx_list:
                idx_list[node.op_type][node.name] = node_nlist.index(node.name)
            else:
                idx_list[node.op_type] = {
                    node.name: node_nlist.index(node.name)
                }
        for op in idx_list:
            print(op)
            count = 0
            for idx in idx_list[op].values():
                count += 1
                print(op,idx,": {")
                temp_list = []
                attri_nlist = []
                # get attribute name list
                #print(graph.node[idx].attribute)
                
                for elem in graph.node[idx].attribute:
                    attri_nlist.append(elem.name)
                    idx1 = attri_nlist.index(elem.name)
                    #print(graph.node[idx].attribute[idx1])
                    ##temp_list.append(graph.node[idx].attribute[idx1])
                
                    if(elem.name == 'group' or elem.name == 'transB' or elem.name == 'axis'):
                        temp_list.append(graph.node[idx].attribute[idx1].i)
                    elif(elem.name == 'max' or elem.name == 'min' or elem.name == 'alpha' or elem.name == 'beta'):
                        temp_list.append(graph.node[idx].attribute[idx1].f)
                    elif(elem.name == 'value'):
                        temp_list.append(graph.node[idx].attribute[idx1].t.raw_data)
                    else:
                        temp_list.append(graph.node[idx].attribute[idx1].ints)
                    #print(attri_nlist)
                    temp_list = ' '.join(map(str, temp_list))
                    print("    ",elem.name,temp_list)
                    
                    temp_list=[]
                    
                print("}")
            print(op,"total :", count)
            print("\n\n")   
    except Exception as e:
        print("Unable to display: " + str(e))
        return False

    return True
```

#### Execution Result
:::info
node數量太多僅貼出部分，會先顯示名字、attribute之後最後顯示layer數量
![](https://course.playlab.tw/md/uploads/aeb19dfd-5ed9-42a5-af2f-eb5065f9bb3b.png)
![](https://course.playlab.tw/md/uploads/dc07a4cd-7fb1-4a0b-ba88-8756a7a08a6c.png)
![](https://course.playlab.tw/md/uploads/97870f41-7318-4273-a20a-1e606f296e09.png)
![](https://course.playlab.tw/md/uploads/1cea8fd8-2d6d-49ed-89f9-7afc2c6b5f8e.png)
![](https://course.playlab.tw/md/uploads/ae7545c8-97b4-422f-ad9b-a5e12d0503f8.png)
![](https://course.playlab.tw/md/uploads/75ff5d40-f01e-4586-8890-aeb91a0ad9e0.png)

:::

### 2-2-2. Data bandwidth requirement 
#### Code
```
def get_valueproto_or_tensorproto_by_name(name: str, graph: xpb2.GraphProto):
    for i, node in enumerate(inferred_model.graph.node):
            if node.name == "":
                inferred_model.graph.node[i].name = str(i)
    input_nlist = [k.name for k in graph.input]
    initializer_nlist = [k.name for k in graph.initializer]
    value_info_nlist = [k.name for k in graph.value_info]
    output_nlist = [k.name for k in graph.output]

    # get tensor data
    if name in input_nlist:
        idx = input_nlist.index(name)
        return graph.input[idx], int(1)
    elif name in value_info_nlist:
        idx = value_info_nlist.index(name)
        return graph.value_info[idx], int(2)
    elif name in initializer_nlist:
        idx = initializer_nlist.index(name)
        return graph.initializer[idx], int(3)
    elif name in output_nlist:
        idx = output_nlist.index(name)
        return graph.output[idx], int(4)
    else:
        print("[ERROR MASSAGE] Can't find the tensor: ", name)
        print('input_nlist:\n', input_nlist)
        print('===================')
        print('value_info_nlist:\n', value_info_nlist)
        print('===================')
        print('initializer_nlist:\n', initializer_nlist)
        print('===================')
        print('output_nlist:\n', output_nlist)
        print('===================')
        return False, 0
def cal_tensor_mem_size(elem_type: str, shape: [int]):
    """ given the element type of the tensor and its shape, and return its memory size.

    Utility.

    Args:
        ttype: the type of the element of the given tensor. format: 'int', ...
        shape: the shape of the given tensor. format: [] of int

    Returns:
        mem_size: int
    """
    # init
    mem_size = int(1)
    # traverse the list to get the number of the elements
    for num in shape:
        mem_size *= num
    # multiple the size of variable with the number of the elements
    # "FLOAT": 1,
    # "UINT8": 2,
    # "INT8": 3,
    # "UINT16": 4,
    # "INT16": 5,
    # "INT32": 6,
    # "INT64": 7,
    # # "STRING" : 8,
    # "BOOL": 9,
    # "FLOAT16": 10,
    # "DOUBLE": 11,
    # "UINT32": 12,
    # "UINT64": 13,
    # "COMPLEX64": 14,
    # "COMPLEX128": 15
    if elem_type == 1:
        mem_size *= 4
    elif elem_type == 2:
        mem_size *= 1
    elif elem_type == 3:
        mem_size *= 1
    elif elem_type == 4:
        mem_size *= 2
    elif elem_type == 5:
        mem_size *= 2
    elif elem_type == 6:
        mem_size *= 4
    elif elem_type == 7:
        mem_size *= 8
    elif elem_type == 9:
        mem_size *= 1
    elif elem_type == 10:
        mem_size *= 2
    elif elem_type == 11:
        mem_size *= 8
    elif elem_type == 12:
        mem_size *= 4
    elif elem_type == 13:
        mem_size *= 8
    elif elem_type == 14:
        mem_size *= 8
    elif elem_type == 15:
        mem_size *= 16
    else:
        print("Undefined data type")

    return mem_size
def get_bandwidth(graph: xpb2.GraphProto):
    try:
        mem_BW_list = []
        total_mem_BW = 0
        unknown_tensor_list = []
        # traverse all the nodes
        for nodeProto in graph.node:
            # init variables
            read_mem_BW_each_layer = 0
            write_mem_BW_each_layer = 0
            total_each_layer = 0
            # traverse all input tensor
            for input_name in nodeProto.input:
                # get the TensorProto/ValueInfoProto by searching its name
                proto, type_Num = get_valueproto_or_tensorproto_by_name(
                    input_name, graph)
                # parse the ValueInfoProto/TensorProto
                if proto:
                    if type_Num == 3:
                        dtype = getattr(proto, 'data_type', False)
                        # get the shape of the tensor
                        shape = getattr(proto, 'dims', [])
                    elif type_Num == 1 or type_Num == 2:
                        name, dtype, shape_str = _parse_element_(proto)
                        if(len(shape_str)>2):#改了這裡
                            shape_str = shape_str.strip('[]')
                            shape_str = shape_str.split(',')
                            shape = []
                            for dim in shape_str:
                                shape.append(int(dim))
                    else:
                        print(
                            '[ERROR MASSAGE] [get_info/mem_BW_without_buf] The Tensor: ',
                            input_name, ' is from a wrong list !')
                else:
                    print(
                        '[ERROR MASSAGE] [get_info/mem_BW_without_buf] The Tensor: ',
                        input_name, ' is no found !')
                    unknown_tensor_list.append(
                        (nodeProto.name, input_name, nodeProto.op_type))
                # calculate the tensor size in btye
                
                read_mem_BW_each_layer += cal_tensor_mem_size(dtype, shape)
    
            # traverse all output tensor
            for output_name in nodeProto.output:
                # get the TensorProto/ValueInfoProto by searching its name
                proto, type_Num = get_valueproto_or_tensorproto_by_name(
                    output_name, graph)
                # parse the ValueInfoProto
                if proto:
                    if type_Num == 2 or type_Num == 4:
                        # name, dtype, shape = utils._parse_ValueInfoProto(proto)
                        name, dtype, shape_str = _parse_element_(proto)
                        
                        if(len(shape_str)>2):#改了這裡
                            shape_str = shape_str.strip('[]')
                            shape_str = shape_str.split(',')
                            shape = []
                            for dim in shape_str:
                                shape.append(int(dim))
                    else:
                        print(
                            '[ERROR MASSAGE] [get_info/mem_BW_without_buf] The Tensor: ',
                            output_name, ' is from a wrong list !')
                else:
                    print(
                        '[ERROR MASSAGE] [get_info/mem_BW_without_buf] The Tensor: ',
                        input_name, ' is no found !')
                    unknown_tensor_list.append(
                        (nodeProto.name, output_name, nodeProto.op_type))
                # calculate the tensor size in btye
                write_mem_BW_each_layer += cal_tensor_mem_size(dtype, shape)
    
            # cal total bw
            total_each_layer = read_mem_BW_each_layer + write_mem_BW_each_layer
    
            # store into tuple
            temp_tuple = (nodeProto.name, read_mem_BW_each_layer,
                        write_mem_BW_each_layer, total_each_layer)
            #append it
            mem_BW_list.append(temp_tuple)
            # accmulate the value
            total_mem_BW += total_each_layer
    
        # display the mem_bw of eahc layer
        columns = ['layer', 'read_bw', 'write_bw', 'total_bw']
        # resort the list
        mem_BW_list = sorted(mem_BW_list,
                                key=lambda Layer: Layer[1],
                                reverse=True)
        #print(tabulate(mem_BW_list, headers=columns))
        print(
            '====================================================================================\n'
        )
        # display it
        print("2-2-2. Data bandwidth requirement\n")
        
        print(
            "The memory bandwidth for processor to execute a whole model without on-chip-buffer is: \n",
            total_mem_BW, '(bytes)\n',
            float(total_mem_BW) / float(1000000), '(MB)\n')
        # display the unknown tensor
        columns = ['op_name', 'unfound_tensor', 'op_type']
        #print(tabulate(unknown_tensor_list, headers=columns))
        print(
            '====================================================================================\n'
        )
    except Exception as e:
        print("[ERROR MASSAGE] Unable to display: " + str(e))
        return False

    return True
```

#### Execution Result
:::info
![](https://course.playlab.tw/md/uploads/2e1b1c9e-dfae-4ac7-932f-01c330f0a4ed.png) 
:::

### 2-2-3. activation memory storage requirement
#### Code
```
def get_act(graph: xpb2.GraphProto):
    print("2-2-3. activation memory storage requirement\n")
    for i, node in enumerate(inferred_model.graph.node):
        if node.name == "":
            inferred_model.graph.node[i].name = str(i)
        # get the list
    All_Conv_tensor_size = []
    Other_type_tensor_size = []
    bias = False
    # get the idx of the operators
    if type(graph) is not xpb2.GraphProto:
        sys.exit('The input graph is not a GraphProto!')

    node_nList = [k.name for k in graph.node]
    input_nlist = [k.name for k in graph.input]
    initializer_nlist = [k.name for k in graph.initializer]
    value_info_nlist = [k.name for k in graph.value_info]
    output_nlist = [k.name for k in graph.output]
    idx_list = {}
    for node in graph.node:
        if node.op_type in idx_list:
            idx_list[node.op_type][node.name] = node_nList.index(node.name)
        else:
            idx_list[node.op_type] = {
                node.name: node_nList.index(node.name)
            }
    #print(graph.input)
    #print(graph.value_info)
    #print(graph.value_info[1])
    #print(graph.output)
    total_activation = 0
    for key in idx_list.keys():
        #print(key,":")
        for idx in idx_list[key].values():
            # temp_tuple, bias = utils._Cal_tensor_size_ConvOrGemm(idx, graph)
            num_conv_input_tensor = len(graph.node[idx].input)
            list_of_data_num = []
            # get input tensor proto
            for input_name in graph.node[idx].input:
                # get tensor data
                if input_name in input_nlist:
                    name_idx = input_nlist.index(input_name)
                    data = graph.input[name_idx]
                    type_num = int(1)
                elif input_name in value_info_nlist:
                    name_idx = value_info_nlist.index(input_name)
                    data = graph.value_info[name_idx]
                    type_num = int(2)
                elif input_name in initializer_nlist:
                    name_idx = initializer_nlist.index(input_name)
                    data = graph.initializer[name_idx]
                    type_num = int(3)
                elif input_name in output_nlist:
                    name_idx = output_nlist.index(input_name)
                    data = graph.output[name_idx]
                    type_num = int(4)
                else:
                    print("Can't find the tensor: ", input_name)
                    print('input_nlist:\n', input_nlist)
                    print('===================')
                    print('value_info_nlist:\n', value_info_nlist)
                    print('===================')
                    print('initializer_nlist:\n', initializer_nlist)
                    print('===================')
                    print('output_nlist:\n', output_nlist)
                    print('===================')
            
                list_of_data_num.append((data, type_num))

            if graph.node[idx].output[0] in input_nlist:
                name_idx = input_nlist.index(graph.node[idx].output[0])
                data = graph.input[name_idx]
                type_num = int(1)
            elif graph.node[idx].output[0] in value_info_nlist:
                name_idx = value_info_nlist.index(graph.node[idx].output[0])
                data = graph.value_info[name_idx]
                type_num = int(2)
            elif graph.node[idx].output[0] in initializer_nlist:
                name_idx = initializer_nlist.index(graph.node[idx].output[0])
                data = graph.initializer[name_idx]
                type_num = int(3)
            elif graph.node[idx].output[0] in output_nlist:
                name_idx = output_nlist.index(graph.node[idx].output[0])
                data = graph.output[name_idx]
                type_num = int(4)
            else:
                print("Can't find the tensor: ", graph.node[idx].output[0])
                print('input_nlist:\n', input_nlist)
                print('===================')
                print('value_info_nlist:\n', value_info_nlist)
                print('===================')
                print('initializer_nlist:\n', initializer_nlist)
                print('===================')
                print('output_nlist:\n', output_nlist)
                print('===================')
            list_of_data_num.append((data, type_num))

            list_temp = [
                graph.node[idx].name,
            ]
            for elem in list_of_data_num:
                if elem[0]:
                    if elem[1] == 3:
                        name = getattr(elem[0], 'name', "None")
                        # get the data type of the tensor
                        data_type = getattr(elem[0], 'data_type', False)
                        # get the shape of the tensor
                        shape = getattr(elem[0], 'dims', [])
                        #print("shape1:",name,shape)
                        continue
                    else:
                        # name, data_type, shape = utils._parse_ValueInfoProto(elem[0])
                        name, data_type, shape_str = _parse_element_(elem[0])
                        
                        
                        if(len(shape_str)>2):
                            shape_str = shape_str.strip('[]')
                            shape_str = shape_str.split(',')
                            shape = []
                            for dim in shape_str:
                                shape.append(int(dim))
                    #print("shape",shape)
                    mem_size = int(1)
                    # traverse the list to get the number of the elements
                    
                    for num in range(1,len(shape)):
                        mem_size *= shape[num]
                    # multiple the size of variable with the number of the elements
                    # "FLOAT": 1,
                    # "UINT8": 2,
                    # "INT8": 3,
                    # "UINT16": 4,
                    # "INT16": 5,
                    # "INT32": 6,
                    # "INT64": 7,
                    # # "STRING" : 8,
                    # "BOOL": 9,
                    # "FLOAT16": 10,
                    # "DOUBLE": 11,
                    # "UINT32": 12,
                    # "UINT64": 13,
                    # "COMPLEX64": 14,
                    # "COMPLEX128": 15
                    if data_type == 1:
                        mem_size *= 4
                    elif data_type == 2:
                        mem_size *= 1
                    elif data_type == 3:
                        mem_size *= 1
                    elif data_type == 4:
                        mem_size *= 2
                    elif data_type == 5:
                        mem_size *= 2
                    elif data_type == 6:
                        mem_size *= 4
                    elif data_type == 7:
                        mem_size *= 8
                    elif data_type == 9:
                        mem_size *= 1
                    elif data_type == 10:
                        mem_size *= 2
                    elif data_type == 11:
                        mem_size *= 8
                    elif data_type == 12:
                        mem_size *= 4
                    elif data_type == 13:
                        mem_size *= 8
                    elif data_type == 14:
                        mem_size *= 8
                    elif data_type == 15:
                        mem_size *= 16
                    
                    list_temp.append(mem_size)
                    #print("list_tmp",list_temp)
                else:
                    print(graph.node[idx].name, 'tenosr no found ! Something wrong')
                    
            if("_0" in list_temp[0]):
                print("input size:", list_temp[1])
                total_activation += list_temp[1]
            if("Constant" not in list_temp[0] and "Clip" not in list_temp[0] and "GlobalAveragePool"not in list_temp[0] and "Add" not in list_temp[0] \
               and "hape" not in list_temp[0] and "Unsqueeze" not in list_temp[0] and "Concat" not in list_temp[0] and "Gather" not in list_temp[0]):
                print(list_temp[0],":", list_temp[2])
                total_activation += list_temp[2]
    print("activation:{} bytes".format(total_activation))
```

#### Execution Result
:::info
![](https://course.playlab.tw/md/uploads/744fcddd-d4a0-4da2-92bf-dde7b9816d61.png)
:::

## HW 2-3 Build tool scripts to manipulate an ONNX model graph

### 2-3-1. create a subgraph (1) that consist of a single Linear layer of size MxKxN

#### Code
```
def create_model_1():
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [128,128])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [128,128])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [128,128])
    
    fc1 = helper.make_node(
        'Gemm',
        inputs=['a', 'b','bias'],
        outputs=['c'],
        alpha=1.0,
        beta=1.0,
        transB=1
    )


    graph = helper.make_graph(
        [fc1],
        'FC',
        [a, b,bias],
        [helper.make_tensor_value_info('c', TensorProto.FLOAT, [128,128])],
        #value_info=[helper.make_tensor_value_info('h1', TensorProto.FLOAT, [1,3])],
        #initializer=[weights1]
    )

    model = helper.make_model(graph, producer_name='hw2-3-1')
    return model
    
model_1 = create_model_1()
onnx.save(model_1, 'hw2-3-1.onnx')
check_model_1 = onnx.load("hw2-3-1.onnx",load_external_data = False)
onnx.checker.check_model(check_model_1)
```

#### Visualize the subgraph (1)
:::info
![](https://course.playlab.tw/md/uploads/2e44dc2f-8391-4e7f-9734-95fb340af1bf.png) 
:::

### 2-3-2. create a subgraph (2) as shown in the above diagram for the subgraph (1)  

#### Code
```
def create_model_2():
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [128,128])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [128,128])
    bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [128,128])
    transpose_node = helper.make_node(
        'Transpose', 
        inputs=['b'], 
        outputs=['bT'], 
        perm=[1, 0]
    )
    split_node_a_0 = helper.make_node(
        'Split',
        inputs=['a'],
        outputs=['a0', 'a1'],
        axis=0,
        num_outputs = 2
    )
    split_node_b_0 = helper.make_node(
        'Split',
        inputs=['bT'],
        outputs=['b0', 'b1'],
        axis=0, 
        num_outputs = 2,
    )
    split_node_a_1_0 = helper.make_node(
        'Split',
        inputs=['a0'],
        outputs=['a00', 'a01'],
        axis=1,
        num_outputs = 2,
    )
    split_node_a_1_1 = helper.make_node(
        'Split',
        inputs=['a1'],
        outputs=['a10','a11'],
        axis=1,
        num_outputs = 2,
    )
    split_node_b_1_0 = helper.make_node(
        'Split',
        inputs=['b0'],
        outputs=['b00', 'b01'],
        axis=1, 
        num_outputs = 2,
    )
    split_node_b_1_1 = helper.make_node(
        'Split',
        inputs=['b1'],
        outputs=['b10','b11'],
        axis=1, 
        num_outputs = 2,
    )
    
    matmul_node_c00_0 = helper.make_node(
        'MatMul',
        inputs=['a00', 'b00'],
        outputs=['c00_0']
    )
    matmul_node_c00_1 = helper.make_node(
        'MatMul',
        inputs=['a01', 'b10'],
        outputs=['c00_1']
    )
    matmul_node_c01_0 = helper.make_node(
        'MatMul',
        inputs=['a00', 'b01'],
        outputs=['c01_0']
    )
    matmul_node_c01_1 = helper.make_node(
        'MatMul',
        inputs=['a01', 'b11'],
        outputs=['c01_1']
    )
    matmul_node_c10_0 = helper.make_node(
        'MatMul',
        inputs=['a10', 'b00'],
        outputs=['c10_0']
    )
    matmul_node_c10_1 = helper.make_node(
        'MatMul',
        inputs=['a11', 'b10'],
        outputs=['c10_1']
    )
    matmul_node_c11_0 = helper.make_node(
        'MatMul',
        inputs=['a10', 'b01'],
        outputs=['c11_0']
    )
    matmul_node_c11_1 = helper.make_node(
        'MatMul',
        inputs=['a11', 'b11'],
        outputs=['c11_1']
    )
    add_node00 = helper.make_node(
        'Add',
        inputs=['c00_0', 'c00_1'],
        outputs=['c00']
    )
    add_node01 = helper.make_node(
        'Add',
        inputs=['c01_0', 'c01_1'],
        outputs=['c01']
    )
    add_node10 = helper.make_node(
        'Add',
        inputs=['c10_0', 'c10_1'],
        outputs=['c10']
    )
    add_node11 = helper.make_node(
        'Add',
        inputs=['c11_0', 'c11_1'],
        outputs=['c11']
    )
    concat_node0 = helper.make_node(
        'Concat',
        inputs=['c00', 'c10'],
        outputs=['c0'],
        axis=0
    )
    concat_node1 = helper.make_node(
        'Concat',
        inputs=['c01', 'c11'],
        outputs=['c1'] ,
        axis=0
    )
    concat_node = helper.make_node(
        'Concat',
        inputs=['c0', 'c1'],
        outputs=['c_'],
        axis=1
    )
    bias_node = helper.make_node(
        'Add',
        inputs=['c_', 'bias'],
        outputs=['c'],
    )
    
    graph = helper.make_graph(
        [transpose_node, split_node_a_0, split_node_b_0, split_node_a_1_0,split_node_a_1_1, split_node_b_1_0,split_node_b_1_1,
         matmul_node_c00_0,matmul_node_c00_1,matmul_node_c01_0,matmul_node_c01_1,
         matmul_node_c10_0,matmul_node_c10_1,matmul_node_c11_0,matmul_node_c11_1,
         add_node00, add_node01, add_node10, add_node11,
         concat_node0,concat_node1,concat_node, bias_node],
        'hw2-3-2_Graph',
        [a, b, bias],
        [helper.make_tensor_value_info('c', onnx.TensorProto.FLOAT, [128, 128])]
    )
    model = helper.make_model(graph, producer_name='hw2-3-2')
    return model

model_2 = create_model_2()
onnx.save(model_2, 'hw2-3-2.onnx')
check_model_2 = onnx.load("hw2-3-2.onnx",load_external_data = False)
onnx.checker.check_model(check_model_2)
```

#### Visualize the subgraph (2)
:::info
![](https://course.playlab.tw/md/uploads/e00eeeca-5883-423c-8d86-a115d2d04b91.png)
:::


### 2-3-3. replace the `Linear` layers in the AlexNet with the equivalent subgraphs (2)
#### Code
```
def get_split_size(length):
    num = length //64#2048
    remainder = length % 64#2048
    split_size = []
    for i in range(num):
        split_size.append(64)#2048
    if(remainder != 0):
        num += 1
        split_size.append(remainder)
    return split_size, num
        
    
def build_new_nodes(idx, my_modified_alexnet, M, N, K, input0, input1, input2, output0):
    transpose_node = helper.make_node(
        'Transpose', 
        inputs=[input1], 
        outputs=[str(idx)+'bT'], 
        perm=[1, 0]
    )
    my_modified_alexnet.node.extend([transpose_node])
    split_sizeN, numN = get_split_size(N)
    output_list_a = []
    for i in range(len(split_sizeN)):
        output_list_a.append(str(idx)+'a'+str(i))
    split_node_a = helper.make_node(
        'Split',
        inputs=[input0],
        outputs=output_list_a,
        axis=1,
        #split=split_sizeN,
        num_outputs = numN
    )
    my_modified_alexnet.node.extend([split_node_a])
    output_list_b = []
    for i in range(len(split_sizeN)):
        output_list_b.append(str(idx)+'b'+str(i))
    split_node_b_0 = helper.make_node(
        'Split',
        inputs=[str(idx)+'bT'],
        outputs=output_list_b,
        axis=0,
        #split=split_sizeN,
        num_outputs = numN
    )
    my_modified_alexnet.node.extend([split_node_b_0])
    split_sizeK, numK = get_split_size(K)
    for i in output_list_b:
        output_list_b1 = []
        for j in range(len(split_sizeK)):
            output_list_b1.append(i+'_'+str(j))
        #print(output_list_b1)
        split_node_b_1 = helper.make_node(
            'Split',
            inputs=[i],
            outputs=output_list_b1,
            axis=1,
            #split=split_sizeK,
            num_outputs = numK
        )
        my_modified_alexnet.node.extend([split_node_b_1])

    for i in range(numK):
        zero_data = np.zeros((1, split_sizeK[i]), dtype=np.float32)
        c_name = str(idx)+'c'+str(i)
        zero_tensor = helper.make_tensor(
            name=c_name,
            data_type=TensorProto.FLOAT,
            dims=[1, split_sizeK[i]],
            vals=zero_data.flatten().tolist()
        )
        add_input_names_ = []
        for j in range(numN):
            inputs_a = str(idx)+'a'+str(j)
            inputs_b = str(idx)+'b'+str(j)+'_'+str(i)
            output_c = str(idx)+'c'+str(i)+'_'+str(j)
            
            multi_node = helper.make_node(
                'MatMul',
                inputs=[inputs_a, inputs_b],
                outputs=[output_c]
            )
            add_input_names_.append(output_c)
            my_modified_alexnet.node.extend([multi_node])

        id_1st_node = helper.make_node(
            'Identity', 
            inputs=[str(idx)+'c'+str(i)+'_0'], 
            outputs=[str(idx)+str(i)+'add_result0']
        )
        my_modified_alexnet.node.extend([id_1st_node])
        for k in range(numN-1):
            in0 = str(idx)+str(i)+'add_result'+str(k)
            in1 = str(idx)+'c'+str(i)+'_'+str(k+1)
            add_node = helper.make_node(
                'Add',
                inputs=[in0,in1],
                outputs=[str(idx)+str(i)+'add_result' + str(k+1)],
            )
            my_modified_alexnet.node.extend([add_node])
        final_add_node = helper.make_node(
            'Identity', 
            inputs=[str(idx)+str(i)+'add_result'+str(numN-1)], 
            outputs=[c_name]
        )
        my_modified_alexnet.node.extend([final_add_node])

    output_c = []
    for c in range(numK):
        tmp = str(idx)+'c' + str(c)
        output_c.append(tmp)
    concat_node = helper.make_node(
        'Concat',
        inputs = output_c,
        outputs = [str(idx)+'c_'],
        axis=1
    )
    my_modified_alexnet.node.extend([concat_node])
    add_node_bias = helper.make_node(
        'Add', 
        inputs=[str(idx)+'c_', input2],
        outputs=output0,
    )
    my_modified_alexnet.node.extend([add_node_bias])
    
alexnet_model = onnx.load('alexnet.onnx')
graph_proto = alexnet_model.graph

input_info = graph_proto.input


output_info = graph_proto.output
tile_size = 64
layer_name = []
input_name = []
output_name = []
for node in graph_proto.node:
    if('Gemm' in node.name):
        layer_name.append(node.name)
        input_name.append(node.input)
        output_name.append(node.output)
    #print(output_name[-1])


my_modified_alexnet = onnx.helper.make_graph(
    nodes = [],
    name = "my_modified_alexnet",
    inputs = graph_proto.input,
    outputs = graph_proto.output,
    initializer = graph_proto.initializer
)

for node in graph_proto.node:
    if(node.name == layer_name[0]):
        input0,input1,input2 = input_name[0]
        output0 = output_name[0]
        M = 1
        N = 9216
        K = 4096
        build_new_nodes(0, my_modified_alexnet, M, N, K,input0,input1,input2,output0)
    
    elif(node.name == layer_name[1]):
        input0,input1,input2 = input_name[1]
        output0 = output_name[1]
        M = 1
        N = 4096
        K = 4096
        build_new_nodes(1, my_modified_alexnet, M, N, K,input0,input1,input2,output0)
    elif(node.name == layer_name[2]):
        input0,input1,input2 = input_name[2]
        output0 = output_name[2]
        M = 1
        N = 4096
        K = 1000
        build_new_nodes(2, my_modified_alexnet, M, N, K,input0,input1,input2,output0)
    else:
        my_modified_alexnet.node.extend([node])
        

model_3 = helper.make_model(my_modified_alexnet, producer_name='my_modified_alexnet')

onnx.save(model_3, 'my_modified_alexnet.onnx')
```

#### Visualize the transformed model graph
:::info
因為使用64作為block size切會導致最後node過多圖過大，netron中也可能顯示不出來。以下為2048為size後的圖，第一個為放大第一個linear layer的部分較方便看。下面的則為完整修改後的alexnet
![](https://course.playlab.tw/md/uploads/51ca5d6d-e1e9-490e-b81c-d4c072f324c7.png)

![](https://course.playlab.tw/md/uploads/3a7eeff9-1409-4b14-9332-06a66dd646d7.png)


:::


### 2-3-4. Correctness Verification
#### Code
```python=
my_onnx_session = ort.InferenceSession("my_modified_alexnet.onnx")
onnx_session = ort.InferenceSession("alexnet.onnx")

input_name = onnx_session.get_inputs()[0].name
input = np.random.rand(10,3,224,224).astype(np.float32)

onnx_output = onnx_session.run(None, {input_name: input})
my_onnx_output = my_onnx_session.run(None, {input_name: input})

print(np.allclose(np.array(my_onnx_output), np.array(onnx_output), atol = 1e-6))
print(np.array(my_onnx_output).shape)
```

#### Execution Result
:::info
![](https://course.playlab.tw/md/uploads/7cb4aab4-07d6-4ad2-a4b4-88035ee19f46.png)
:::



## HW 2-4 Using Pytorch C++ API to do model analysis on the transformed model graph

這部份我先用pytorch定義好一個新的MyLinear node來取代本來的Linear，之後生成alexnet，產生script後用C++ tool計算需要的數據，程式如下:
```python=
class MyLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    def get_split_size_(self, length: int, BLOCK_SIZE: int) -> List[int]:
        num = length // BLOCK_SIZE
        remainder = length % BLOCK_SIZE
        split_size = []
        for i in range(num):
            split_size.append(BLOCK_SIZE)
        if(remainder != 0):
            num += 1
            split_size.append(remainder)
        return split_size
    def forward(self, input: Tensor) -> Tensor:
        M = input.size(0)
        N = self.in_features
        K = self.out_features
        split_size = self.get_split_size_(K, BLOCK_SIZE)
        a_split = torch.split(input, split_size_or_sections=BLOCK_SIZE, dim=1)
        b = self.weight
        bias = self.bias
        b_dim1 = torch.split(b, split_size_or_sections=BLOCK_SIZE, dim=1)
        b_blk = []
        for b_0 in b_dim1:
            b_dim0 = torch.split(b_0, split_size_or_sections=BLOCK_SIZE, dim=0)
            b_blk.append(b_dim0)
        
        c_blk = []
        for i in range(len(b_blk)):
            tmp_matrix = torch.zeros(M, split_size[i])
            for j in range(len(b_blk[i])):
                tmp_matrix += a_split[j] @ b_blk[i][j]
            c_blk.append(tmp_matrix)
        
        c = torch.cat(c_blk, dim=1)
        c += bias 
        return c

class modified_AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.maxpool1 =  nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.maxpool2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.maxpool3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(1,-1)
        )
        
        #classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            MyLinear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            MyLinear(4096, 4096),
            nn.ReLU(inplace=True),
            MyLinear(4096, 1000)
        )        
    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x=self.conv1(x)
        x=self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x= self.conv5(x)
        x = self.maxpool3(x)
        x = self.avgpool(x)
        x = self.classifier(x)        
        return x
modified_model = modified_AlexNet(num_classes=NUM_classes)
input_data = torch.randn(1, 3, 224, 224)
traced_script_module = torch.jit.trace(modified_model, input_data)
traced_script_module.save("tracedmodified_alexnet_new.pt")
```

### 2-4-1. Calculate memory requirements for storing the model weights.
### 2-4-2. Calculate memory requirements for storing the activations
### 2-4-3. Calculate computation requirements
#### Code
這三個part程式、執行畫面一起放，在下面部分都有呈現。
```cpp=
void dump_to_str2_4(const torch::jit::script::Module& module, size_t& total_param_size, int &total_macs,vector<int> &inputsize, vector<vector<int>> &layer_output_shapes, int &idx){
    //std::stringstream ss;
    
    int in_w=1, out_w=1, in_h=1, out_h=1;
    if(idx == 0){
        in_w = inputsize[1]; in_h = inputsize[2];
        out_w = layer_output_shapes[idx][2]; out_h = layer_output_shapes[idx][3];
    } else if(idx > 0 && idx <=7){
        in_w = layer_output_shapes[idx-1][2]; in_h = layer_output_shapes[idx-1][3];
        out_w = layer_output_shapes[idx][2]; out_h = layer_output_shapes[idx][3];
    } else {in_w=1; out_w=1; in_h=1; out_h=1;}
    
    if ((containsSubstring (module.type()->name()->qualifiedName(), "Linear")) ||(containsSubstring(module.type()->name()->qualifiedName(), "Conv2d"))||(containsSubstring (module.type()->name()->qualifiedName(), "Max"))){
        cout << module.type()->name()->qualifiedName() <<"\n";
        if(containsSubstring (module.type()->name()->qualifiedName(), "Max"))){
            cout<<"activation size: ["<<layer_output_shapes[idx][1]<<" "<<layer_output_shapes[idx][2]<<" "<<layer_output_shapes[idx][3]<<"]"<<"\n";
        }
        for (const auto& p : module.named_parameters(/*recurse=*/false)) {
            
            const torch::Tensor& param_tensor = p.value;
            //std::vector<int64_t> shape = param_tensor.sizes();
            vector<int64_t> shape(param_tensor.sizes().begin(), param_tensor.sizes().end());
            //std::cout<<shape<<"!~!\n";
                        
            if(shape.size()>1){
                size_t param_size = 1;
                for (int64_t dim : shape) {
                    param_size *= dim;
                }
                
                caffe2::TypeMeta dtype = param_tensor.dtype();
            
                // 根據數據類型確定每個值的大小
                size_t element_size = 0;
                if (dtype == caffe2::TypeMeta::Make<float>()) {
                    element_size = sizeof(float);
                } else if (dtype == caffe2::TypeMeta::Make<double>()) {
                    element_size = sizeof(double);
                } else if (dtype == caffe2::TypeMeta::Make<int32_t>()) {
                    element_size = sizeof(int32_t);
                }

                if(containsSubstring (module.type()->name()->qualifiedName(), "Linear")){
                    cout<<"activation size: ["<<shape[1]<<",]"<<"\n";
                    int macs = param_size * out_w * out_h;
                    param_size += shape[1];
                    cout <<"macs:"<<macs<<"\n"<<"para #"<<param_size<<"\n";
                    total_macs += macs;
                    total_param_size += param_size * element_size;
                }else if(containsSubstring (module.type()->name()->qualifiedName(), "Conv2d")){
                    cout<<"activation size: ["<<shape[0]<<" "<<out_w<<" "<<out_h<<"]"<<"\n";
                    int macs = param_size * out_w * out_h;
                    param_size += shape[0];
                    cout <<"macs:"<<macs<<"\n"<<"para #"<<param_size<<"\n";
                    total_macs += macs;
                    total_param_size += param_size * element_size;
                }
                //total_macs += (param_size * out_w * out_h);
                //total_param_size += param_size * element_size;
            }
        }
        ++idx;
    }

    for (const auto& s : module.named_children()) {
        dump_to_str2_4(s.value, total_param_size, total_macs,inputsize, layer_output_shapes, idx);
    }
    
}

dump_to_str2_4(module,total_param_size,total_macs, inputsize, layer_output_shapes,idx);
cout << "total_param_size:" << total_param_size << endl;
cout << "total_macs:" << total_macs << endl;

```

#### Execution Result
:::info
![](https://course.playlab.tw/md/uploads/11d3cfcc-647f-4145-bb3d-7dd6d1cb3a81.png)
:::




### 2-4-4. Compare your results to the result in HW2-1 and HW2-2

#### Discussion
:::info
Lab2-1中原來的alexnet 用python套件算出來Total memory for parameters:  244403360
macs: 714188480
還有印出的activation memory也是一樣，
和我這裡算出來的相符，應該是沒有問題的。
:::


## Others
- If you have any comment or recommendation to this lab, you can write it down here to tell us.
希望以後可以早點多點提示，例如說2-3要使用make_node這類之前lab沒出現的工具，當時研究了很久才確定要用這個工具，或2-4當時一開始也是一頭霧水。
整體講時間很緊急也學到很多東西。也讓我深刻體認自身不足，有些部分我可能做的還不夠好，我會慢慢把沒做好的地方訂正完善

