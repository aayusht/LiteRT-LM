#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_ENGINE_ADVANCED_LEGACY_IMPL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_ENGINE_ADVANCED_LEGACY_IMPL_H_

#include <memory>
#include <optional>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "third_party/odml/infra/genai/inference/executor/litert_executor_utils.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/framework/resource_management/execution_manager.h"

namespace litert::lm {

class EngineAdvancedLegacyImpl : public Engine {
 public:
  ~EngineAdvancedLegacyImpl() override;

  static absl::StatusOr<std::unique_ptr<Engine>> CreateEngine(
      EngineSettings engine_settings);

  absl::StatusOr<std::unique_ptr<Session>> CreateSession(
      const SessionConfig& session_config) const override;

  absl::Status WaitUntilDone(absl::Duration timeout) override;

  const EngineSettings& GetEngineSettings() const override;

 private:
  explicit EngineAdvancedLegacyImpl(
      EngineSettings engine_settings,
      std::unique_ptr<odml::infra::ExecutorModelResources> model_resources,
      std::unique_ptr<ExecutionManager> execution_manager,
      Tokenizer* absl_nonnull tokenizer,
      std::unique_ptr<Tokenizer> task_tokenizer,
      std::optional<BenchmarkInfo> benchmark_info);

  EngineSettings engine_settings_;
  std::unique_ptr<odml::infra::ExecutorModelResources> model_resources_;
  std::shared_ptr<ExecutionManager> execution_manager_;
  Tokenizer* tokenizer_;
  std::unique_ptr<Tokenizer> task_tokenizer_;
  std::optional<BenchmarkInfo> benchmark_info_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_ENGINE_ADVANCED_LEGACY_IMPL_H_
