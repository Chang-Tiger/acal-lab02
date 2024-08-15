#include <torch/script.h> // One-stop header.
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/ir/ir.h>

#include <iostream>
#include <memory>
using namespace std;
using namespace torch::jit;
bool containsSubstring(const string& str, const string& substr) {
    return str.find(substr) != string::npos;
}


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



int main(int argc, const char* argv[]) {
    if (argc != 2) {
        cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }
    
    set_jit_logging_levels("GRAPH_DUMP");
    
    torch::jit::script::Module module;
    try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        cerr << "error loading the model\n";
        return -1;
    }
    
    cout << "load the torchscript model, " + string(argv[1]) + ", successfully \n";
    
    // Create a vector of inputs.
    vector<torch::jit::IValue> inputs;
    vector<vector<int>> layer_output_shapes;
    inputs.push_back(torch::ones({1, 3, 224, 224}));
    vector<int> inputsize = {3, 244, 244};
    for(const auto& subm : module.children()){
        auto& mutable_subm = const_cast<torch::jit::Module&>(subm);
        at::Tensor output = mutable_subm.forward(inputs).toTensor();
        //cout<<output.sizes()<<endl;
        vector<int> temp(output.sizes().begin(),output.sizes().end());
        layer_output_shapes.push_back(temp);
        inputs[0] = output;
    }
    cout<<layer_output_shapes[1]<<endl;
    
    // https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/api/module.cpp
    cout << "dump the model information" << " (\n";
    int total_macs = 0;
    size_t total_param_size = 0;
    int idx = 0;
    dump_to_str2_4(module,total_param_size,total_macs, inputsize, layer_output_shapes,idx);
    cout << "total_param_size:" << total_param_size << endl;
    cout << "total_macs:" << total_macs << endl;

  
}
