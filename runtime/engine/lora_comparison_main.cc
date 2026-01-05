// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "base/init_google.h"
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"

ABSL_FLAG(std::string, model_path, "", "Path to the base model tflite file.");
ABSL_FLAG(std::string, lora_path, "", "Path to the LoRA weights tflite file.");
ABSL_FLAG(int, lora_rank, 0, "LoRA rank. If 0, lora_path is ignored.");
ABSL_FLAG(std::string, backend, "cpu", "Backend to use.");
ABSL_FLAG(
    std::string, prompt,
    "Summarize this article in one sentence: Scientists have developed a new "
    "type of battery that uses sodium-ion technology, offering a potentially "
    "cheaper and more sustainable alternative to lithium-ion batteries. While "
    "lithium is effective, its extraction is costly and geographically "
    "concentrated. The new sodium-ion batteries show promising performance in "
    "terms of energy density and cycle life, though they are not yet on par "
    "with the best lithium-ion cells. Researchers believe that with further "
    "development, sodium-ion could become a viable option for grid storage "
    "and electric vehicles, reducing reliance on lithium.",
    "Input prompt to run inference on.");

namespace litert::lm {

absl::StatusOr<std::string> RunInference(
    EngineSettings engine_settings, absl::string_view prompt,
    std::optional<std::string> lora_path = std::nullopt) {
  ASSIGN_OR_RETURN(auto engine,
                   Engine::CreateEngine(std::move(engine_settings)));
  SessionConfig session_config = SessionConfig::CreateDefault();
  if (lora_path.has_value()) {
    ASSIGN_OR_RETURN(auto lora_file, litert::lm::ScopedFile::Open(*lora_path));
    session_config.SetScopedLoraFile(
        std::make_shared<litert::lm::ScopedFile>(std::move(lora_file)));
  }
  ASSIGN_OR_RETURN(auto session, engine->CreateSession(session_config));

  std::vector<InputData> inputs;
  inputs.push_back(InputText(std::string(prompt)));

  RETURN_IF_ERROR(session->RunPrefill(inputs));
  ASSIGN_OR_RETURN(auto responses, session->RunDecode());

  if (responses.GetTexts().empty() || responses.GetTexts()[0].empty()) {
    return absl::InternalError("Received an empty response from decoder.");
  }
  return responses.GetTexts()[0];
}

absl::Status Run() {
  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  const std::string lora_path = absl::GetFlag(FLAGS_lora_path);
  const int lora_rank = absl::GetFlag(FLAGS_lora_rank);
  const std::string prompt = absl::GetFlag(FLAGS_prompt);
  const std::string backend_str = absl::GetFlag(FLAGS_backend);
  ASSIGN_OR_RETURN(Backend backend, GetBackendFromString(backend_str));

  if (model_path.empty()) {
    return absl::InvalidArgumentError("--model_path must be specified.");
  }
  if (lora_rank > 0 && lora_path.empty()) {
    return absl::InvalidArgumentError(
        "--lora_path must be specified if --lora_rank > 0.");
  }

  // Run 1: Without LoRA
  ASSIGN_OR_RETURN(ModelAssets model_assets_no_lora,
                   ModelAssets::Create(model_path));
  ASSIGN_OR_RETURN(
      EngineSettings engine_settings_no_lora,
      EngineSettings::CreateDefault(std::move(model_assets_no_lora), backend));
  engine_settings_no_lora.GetMutableMainExecutorSettings().SetLoraRank(0);

  std::cout << "\n--- Running inference WITHOUT LoRA ---\n";
  ASSIGN_OR_RETURN(std::string result_no_lora,
                   RunInference(std::move(engine_settings_no_lora), prompt));
  std::cout << "Output without LoRA:\n" << result_no_lora << "\n";

  // Run 2: With LoRA
  if (lora_rank > 0) {
    ASSIGN_OR_RETURN(ModelAssets model_assets_with_lora,
                     ModelAssets::Create(model_path));
    ASSIGN_OR_RETURN(
        EngineSettings engine_settings_with_lora,
        EngineSettings::CreateDefault(std::move(model_assets_with_lora), backend));
    engine_settings_with_lora.GetMutableMainExecutorSettings().SetLoraRank(
        lora_rank);
    RETURN_IF_ERROR(engine_settings_with_lora.GetMutableMainExecutorSettings()
                        .SetSupportedLoraRanks(
                            {static_cast<unsigned int>(lora_rank)}));

    std::cout << "\n--- Running inference WITH LoRA (rank=" << lora_rank
              << ") ---\n";
    ASSIGN_OR_RETURN(
        std::string result_with_lora,
        RunInference(std::move(engine_settings_with_lora), prompt, lora_path));
    std::cout << "Output with LoRA:\n" << result_with_lora << "\n";

    if (result_no_lora == result_with_lora) {
      std::cout << "\nResult: LoRA produced the same output as base model.\n";
    } else {
      std::cout << "\nResult: LoRA produced a different output.\n";
    }
  }

  return absl::OkStatus();
}

}  // namespace litert::lm

int main(int argc, char* argv[]) {
  InitGoogle(argv[0], &argc, &argv, true);
  ABSL_CHECK_OK(litert::lm::Run());
  return 0;
}
