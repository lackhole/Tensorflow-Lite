
* Change Macro for using GPU or NNAPI delegates (or it won't compile)  
https://github.com/visualcamp/Tensorflow-Lite/blob/cutemodel/CuteModel.hpp#L8

* Header
```
#include "cutemodel/CuteModel.hpp"
```

* Building a model
```
ct::CuteModel model;
model = ct::CuteModel(buffer, buffer_size); // build from buffer
// model = ct::CuteModel("models/my_custom_model.tflite"); build from file
model.setGpuDelegate(); // option
model.setCpuNumThreads(2); // if using gpu delegate, setting threads is meaningless
model.buildInterpreter();

if(model.isBuilt(){
    std::string info = model.summary();
    std::string info_options = model.summarizeOptions();
}
```

* Set Input
```
int data_1 = { ... };
double *data_2 = { ... };
std::vector<float> data_3 = { ... };

model.setInput(&data_1, data_2, data_3.data());
```

* Run an inference
```
model.invoke();
```

* Get output

  * `copyOutputToBuffer`
    * buffer MUST be allocated
  ```
  std::vector<float> out(10);
  model.copyOutputToBuffer(0, out.data());  // #1
  model.copyOutputToBuffer(0, out.data(), sizeof(float)*10); // #2
  ```

  * `getOutput`
  ```
  std::vector<float> out = model.getOutput<float>(0);

  // deprecated
  // model.getOutput(0, out);
  ```
