/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <iostream>
#include <limits>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/core/category.h"
#include "tensorflow_lite_support/cc/task/text/nlclassifier/nl_classifier.h"

ABSL_FLAG(std::string, model_path, "",
          "Absolute path to the '.tflite' classification model.");
ABSL_FLAG(std::string, text, "", "Text to classify.");
ABSL_FLAG(int, input_tensor_index, -1, "Input tensor index of the model.");
ABSL_FLAG(int, output_score_tensor_index, -1,
          "Output score tensor index of the model.");
ABSL_FLAG(int, output_label_tensor_index, -1,
          "Output label tensor index of the model.");
ABSL_FLAG(std::string, input_tensor_name, "",
          "Input tensor name of the model.");
ABSL_FLAG(std::string, output_score_tensor_name, "",
          "Output score tensor name of the model.");
ABSL_FLAG(std::string, output_label_tensor_name, "",
          "Output label tensor name of the model.");

namespace tflite {
namespace task {
namespace text {
namespace nlclassifier {

absl::Status Classify() {
  NLClassifierOptions options{};
  if (absl::GetFlag(FLAGS_input_tensor_index) >= 0) {
    options.input_tensor_index = absl::GetFlag(FLAGS_input_tensor_index);
  }
  if (absl::GetFlag(FLAGS_output_score_tensor_index) >= 0) {
    options.output_score_tensor_index =
        absl::GetFlag(FLAGS_output_score_tensor_index);
  }
  if (absl::GetFlag(FLAGS_output_label_tensor_index) >= 0) {
    options.output_label_tensor_index =
        absl::GetFlag(FLAGS_output_label_tensor_index);
  }
  if (!absl::GetFlag(FLAGS_input_tensor_name).empty()) {
    options.input_tensor_name = absl::GetFlag(FLAGS_input_tensor_name);
  }
  if (!absl::GetFlag(FLAGS_output_score_tensor_name).empty()) {
    options.output_score_tensor_name =
        absl::GetFlag(FLAGS_output_score_tensor_name);
  }
  if (!absl::GetFlag(FLAGS_output_label_tensor_name).empty()) {
    options.output_label_tensor_name =
        absl::GetFlag(FLAGS_output_label_tensor_name);
  }

  ASSIGN_OR_RETURN(std::unique_ptr<NLClassifier> classifier,
                   NLClassifier::CreateFromFileAndOptions(
                       absl::GetFlag(FLAGS_model_path), options));

  std::vector<core::Category> categories =
      classifier->Classify(absl::GetFlag(FLAGS_text));

  for (int i = 0; i < categories.size(); ++i) {
    const core::Category& category = categories[i];
    std::cout << absl::StrFormat("category[%d]: '%s' : '%.5f'\n", i,
                                 category.class_name, category.score);
  }

  return absl::OkStatus();
}

}  // namespace nlclassifier
}  // namespace text
}  // namespace task
}  // namespace tflite

int main(int argc, char** argv) {
  // Parse command line arguments and perform sanity checks.
  absl::ParseCommandLine(argc, argv);
  if (absl::GetFlag(FLAGS_model_path).empty()) {
    std::cerr << "Missing mandatory 'model_path' argument.\n";
    return 1;
  }
  if (absl::GetFlag(FLAGS_text).empty()) {
    std::cerr << "Missing mandatory 'question' argument.\n";
    return 1;
  }

  // Run classification.
  absl::Status status = tflite::task::text::nlclassifier::Classify();
  if (status.ok()) {
    return 0;
  } else {
    std::cerr << "Classification failed: " << status.message() << "\n";
    return 1;
  }
}
