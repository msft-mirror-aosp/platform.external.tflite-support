/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/op_resolver.h"

namespace tflite {
namespace task {

// Create a minimal MutableOpResolver to provide only
// the ops required by NLClassifier/BertNLClassifier.
std::unique_ptr<MutableOpResolver> CreateOpResolver() {
  MutableOpResolver resolver;
  resolver.AddBuiltin(::tflite::BuiltinOperator_DEQUANTIZE,
                      ::tflite::ops::builtin::Register_DEQUANTIZE());
  resolver.AddBuiltin(::tflite::BuiltinOperator_RESHAPE,
                      ::tflite::ops::builtin::Register_RESHAPE());
  resolver.AddBuiltin(::tflite::BuiltinOperator_GATHER,
                      ::tflite::ops::builtin::Register_GATHER());
  resolver.AddBuiltin(::tflite::BuiltinOperator_STRIDED_SLICE,
                      ::tflite::ops::builtin::Register_STRIDED_SLICE());
  resolver.AddBuiltin(::tflite::BuiltinOperator_PAD,
                      ::tflite::ops::builtin::Register_PAD());
  resolver.AddBuiltin(::tflite::BuiltinOperator_CONCATENATION,
                      ::tflite::ops::builtin::Register_CONCATENATION());
  resolver.AddBuiltin(::tflite::BuiltinOperator_FULLY_CONNECTED,
                      ::tflite::ops::builtin::Register_FULLY_CONNECTED());
  resolver.AddBuiltin(::tflite::BuiltinOperator_CAST,
                      ::tflite::ops::builtin::Register_CAST());
  resolver.AddBuiltin(::tflite::BuiltinOperator_MUL,
                      ::tflite::ops::builtin::Register_MUL());
  resolver.AddBuiltin(::tflite::BuiltinOperator_ADD,
                      ::tflite::ops::builtin::Register_ADD());
  resolver.AddBuiltin(::tflite::BuiltinOperator_TRANSPOSE,
                      ::tflite::ops::builtin::Register_TRANSPOSE());
  resolver.AddBuiltin(::tflite::BuiltinOperator_SPLIT,
                      ::tflite::ops::builtin::Register_SPLIT());
  resolver.AddBuiltin(::tflite::BuiltinOperator_PACK,
                      ::tflite::ops::builtin::Register_PACK());
  resolver.AddBuiltin(::tflite::BuiltinOperator_SOFTMAX,
                      ::tflite::ops::builtin::Register_SOFTMAX());
  return absl::make_unique<MutableOpResolver>(resolver);
}

}  // namespace task
}  // namespace tflite