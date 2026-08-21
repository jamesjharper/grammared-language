"""Tests for config_parser utility module."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import yaml

from grammared_language.utils.config_parser import (
    load_config_from_file,
    create_client_from_config,
    create_clients_from_config,
    ModelsConfig,
    ModelClientInitializationError,
    GectorConfig,
    GrammaredClassifierConfig,
    CoEditConfig,
    ServingConfig
)


class TestLoadConfig:
    """Test load_config_from_file function."""
    
    def test_load_config_from_file_success(self, tmp_path):
        """Test loading a valid config file with new nested format."""
        config_data = {
            'model1': {
                'type': 'gector',
                'backend': 'triton',
                'serving_config': {
                    'pretrained_model_name_or_path': 'test-model-1',
                    'triton_model_name': 'gector_model',
                    'triton_hostname': 'localhost',
                    'triton_port': 8001
                }
            },
            'model2': {
                'type': 'grammared_classifier',
                'backend': 'triton',
                'serving_config': {
                    'pretrained_model_name_or_path': 'test-model-2',
                    'triton_model_name': 'classifier_model',
                    'triton_hostname': 'localhost',
                    'triton_port': 8001
                }
            }
        }
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        result = load_config_from_file(str(config_file))
        assert isinstance(result, ModelsConfig)
        assert len(result.models) == 2
        assert 'model1' in result.models
        assert 'model2' in result.models
        assert result.models['model1'].serving_config.pretrained_model_name_or_path == 'test-model-1'
    
    def test_load_config_from_file_file_not_found(self):
        """Test that FileNotFoundError is raised for missing config."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config_from_file("nonexistent_config.yaml")


class TestCreateClientFromConfig:
    """Test create_client_from_config function."""
    
    @patch('grammared_language.clients.gector_client.GectorClient')
    def test_create_gector_client(self, mock_gector):
        """Test creating a GectorClient from new config format."""
        config = {
            'type': 'gector',
            'backend': 'triton',
            'serving_config': {
                'pretrained_model_name_or_path': 'test-model',
                'triton_model_name': 'test_triton_model',
                'triton_hostname': 'localhost',
                'triton_port': 8001
            }
        }
        
        mock_client = Mock()
        mock_gector.return_value = mock_client
        
        result = create_client_from_config('test_model', config)
        
        assert result is not None
        mock_gector.assert_called_once()
        # Verify the parameters passed
        call_kwargs = mock_gector.call_args[1]
        assert call_kwargs['pretrained_model_name_or_path'] == 'test-model'
        assert call_kwargs['triton_model_name'] == 'test_triton_model'

    @patch('grammared_language.clients.gector_client.GectorClient')
    def test_gector_prediction_batch_size_comes_from_inference_config(self, mock_gector):
        """GECToR helper chunking is not coupled to Triton scheduler batching."""
        config = {
            'type': 'gector',
            'serving_config': {
                'pretrained_model_name_or_path': 'test-model',
                'triton_model_name': 'test_triton_model',
                'max_batch_size': 32,
            },
            'model_inference_config': {'batch_size': 3},
        }

        create_client_from_config('test_model', config)

        call_kwargs = mock_gector.call_args[1]
        assert call_kwargs['batch_size'] == 3
        assert 'max_batch_size' not in call_kwargs
    
    @patch('grammared_language.clients.grammar_classification_client.GrammarClassificationClient')
    def test_create_grammared_classifier_client(self, mock_classifier):
        """Test creating a GrammarClassificationClient from new config format."""
        config = {
            'type': 'grammared_classifier',
            'backend': 'triton',
            'serving_config': {
                'pretrained_model_name_or_path': 'test-model',
                'triton_model_name': 'test_triton_model',
                'triton_hostname': 'localhost',
                'triton_port': 8001
            }
        }
        
        mock_client = Mock()
        mock_classifier.return_value = mock_client
        
        result = create_client_from_config('test_model', config)
        
        assert result is not None
        mock_classifier.assert_called_once()
    
    @patch('grammared_language.clients.coedit_client.CoEditClient')
    def test_create_coedit_client(self, mock_coedit):
        """Test creating a CoEditClient from new config format."""
        config = {
            'type': 'coedit',
            'backend': 'triton',
            'serving_config': {
                'pretrained_model_name_or_path': 'test-model',
                'triton_model_name': 'test_triton_model',
                'triton_hostname': 'localhost',
                'triton_port': 8001
            }
        }
        
        mock_client = Mock()
        mock_coedit.return_value = mock_client
        
        result = create_client_from_config('test_model', config)
        
        assert result is not None
        mock_coedit.assert_called_once()
    
    @patch('grammared_language.clients.coedit_client.CoEditClient')
    def test_create_coedit_client_with_prompt_template(self, mock_coedit):
        """Test creating a CoEditClient with grammared_config."""
        config = {
            'type': 'coedit',
            'backend': 'triton',
            'serving_config': {
                'pretrained_model_name_or_path': 'test-model',
                'triton_model_name': 'test_triton_model',
                'triton_hostname': 'localhost',
                'triton_port': 8001
            },
            'grammared_config': {
                'prompt_template': 'Fix grammatical errors: {{ text }}'
            }
        }
        
        mock_client = Mock()
        mock_coedit.return_value = mock_client
        
        result = create_client_from_config('test_model', config)
        
        assert result is not None
        call_kwargs = mock_coedit.call_args[1]
        assert call_kwargs['prompt_template'] == 'Fix grammatical errors: {{ text }}'
    
    def test_create_client_unknown_type(self):
        """Test handling of unknown client type."""
        config = {
            'type': 'unknown_type',
            'backend': 'triton',
            'serving_config': {
                'triton_hostname': 'localhost'
            }
        }
        
        with pytest.raises(ValueError, match="Unknown model type for test_model: unknown_type"):
            create_client_from_config('test_model', config)
    
    @patch('grammared_language.clients.gector_client.GectorClient')
    def test_create_client_propagates_initialization_error(self, mock_gector):
        """Direct client creation must not convert failures into None."""
        config = {
            'type': 'gector',
            'backend': 'triton',
            'serving_config': {
                'pretrained_model_name_or_path': 'test-model',
                'triton_model_name': 'test_triton_model'
            }
        }
        
        mock_gector.side_effect = Exception("Test error")
        
        with pytest.raises(Exception, match="Test error"):
            create_client_from_config('test_model', config)


class TestCreateClientsFromConfig:
    """Test create_clients_from_config function."""
    
    @patch('grammared_language.utils.config_parser.create_client_from_config')
    @patch('grammared_language.utils.config_parser.get_config')
    def test_create_clients_from_config(self, mock_get_config, mock_create):
        """Test creating multiple clients from config file."""
        mock_models_config = ModelsConfig(
            models={
                'model1': GectorConfig(
                    type='gector',
                    backend='triton',
                    serving_config=ServingConfig(
                        pretrained_model_name_or_path='test1',
                        triton_model_name='gector_model'
                    )
                ),
                'model2': GrammaredClassifierConfig(
                    type='grammared_classifier',
                    backend='triton',
                    serving_config=ServingConfig(
                        pretrained_model_name_or_path='test2',
                        triton_model_name='classifier_model'
                    )
                )
            }
        )
        mock_get_config.return_value = mock_models_config
        
        mock_client1 = Mock()
        mock_client2 = Mock()
        mock_create.side_effect = [mock_client1, mock_client2]
        
        result = create_clients_from_config('test_config.yaml')
        
        assert len(result) == 2
        assert result[0] == mock_client1
        assert result[1] == mock_client2
    
    @patch('grammared_language.utils.config_parser.create_client_from_config')
    @patch('grammared_language.utils.config_parser.get_config')
    def test_create_clients_skip_invalid(self, mock_get_config, mock_create):
        """Test that invalid clients are skipped."""
        mock_models_config = ModelsConfig(
            models={
                'model1': GectorConfig(
                    type='gector',
                    backend='triton',
                    serving_config=ServingConfig(
                        pretrained_model_name_or_path='test1',
                        triton_model_name='gector_model'
                    )
                ),
                'model3': GrammaredClassifierConfig(
                    type='grammared_classifier',
                    backend='triton',
                    serving_config=ServingConfig(
                        pretrained_model_name_or_path='test3',
                        triton_model_name='classifier_model'
                    )
                )
            }
        )
        mock_get_config.return_value = mock_models_config
        
        mock_client1 = Mock()
        mock_client3 = Mock()
        mock_create.side_effect = [mock_client1, mock_client3]
        
        result = create_clients_from_config('test_config.yaml')
        
        # Should have 2 clients
        assert len(result) == 2
        assert mock_create.call_count == 2
    
    @patch('grammared_language.utils.config_parser.create_client_from_config')
    @patch('grammared_language.utils.config_parser.get_config')
    def test_configured_client_failure_aborts_startup(self, mock_get_config, mock_create):
        """A configured model cannot disappear from the client set."""
        mock_models_config = ModelsConfig(
            models={
                'model1': GectorConfig(
                    type='gector',
                    backend='triton',
                    serving_config=ServingConfig(
                        pretrained_model_name_or_path='test1',
                        triton_model_name='gector_model'
                    )
                ),
                'model2': CoEditConfig(
                    type='coedit',
                    backend='triton',
                    serving_config=ServingConfig(
                        pretrained_model_name_or_path='test2',
                        triton_model_name='gector_model2'
                    )
                )
            }
        )
        mock_get_config.return_value = mock_models_config
        
        mock_client1 = Mock()
        mock_create.side_effect = [mock_client1, ConnectionError('CoEdIT unavailable')]

        with pytest.raises(ModelClientInitializationError) as raised:
            create_clients_from_config('test_config.yaml')

        message = str(raised.value)
        assert "name='model2'" in message
        assert "type='coedit'" in message
        assert "backend='triton'" in message
        assert "grpc://localhost:8001" in message
        assert "CoEdIT unavailable" in message
        assert isinstance(raised.value.__cause__, ConnectionError)

class TestNestedConfigStructure:
    """Test new nested config structure."""
    
    def test_coedit_config_with_all_nested_configs(self):
        """Test CoEditConfig with all nested configuration sections."""
        config_dict = {
            'type': 'coedit',
            'backend': 'triton',
            'serving_config': {
                'triton_hostname': 'localhost',
                'triton_port': 8001,
                'pretrained_model_name_or_path': 'test-model',
                'triton_model_name': 'coedit',
                'backend': 'ort',
                'device': 'cpu'
            },
            'model_init_config': {
                'load_in_4bit': False
            },
            'model_inference_config': {
                'temperature': 0.8,
                'max_length': 512,
                'do_sample': True,
            },
            'grammared_config': {
                'prompt_template': 'Fix grammatical errors: {{ text }}'
            }
        }
        
        config = CoEditConfig(**config_dict)
        
        assert config.type == 'coedit'
        assert config.backend == 'triton'
        assert config.serving_config.triton_hostname == 'localhost'
        assert config.serving_config.triton_port == 8001
        assert config.serving_config.triton_model_name == 'coedit'
        assert config.model_inference_config.temperature == 0.8
        assert config.model_init_config.load_in_4bit is False
        assert config.grammared_config.prompt_template == 'Fix grammatical errors: {{ text }}'

    @patch('grammared_language.clients.coedit_client.CoEditClient')
    def test_coedit_uses_served_triton_name_not_logical_name(self, mock_coedit):
        config = {
            'type': 'coedit',
            'serving_config': {
                'pretrained_model_name_or_path': 'test-model',
                'triton_model_name': 'actual_triton_name',
            },
        }

        create_client_from_config('my_coedit', config)

        assert mock_coedit.call_args.kwargs['model_name'] == 'actual_triton_name'
        assert mock_coedit.call_args.kwargs['rule_id'] == 'my_coedit'

    @pytest.mark.parametrize(
        ('serving_overrides', 'grammared_overrides', 'expected_served_name', 'expected_rule_id'),
        [
            ({}, {}, 'coedit_large', 'coedit_large'),
            (
                {'triton_model_name': 'coedit_large_v2'},
                {},
                'coedit_large_v2',
                'coedit_large',
            ),
            ({}, {'rule_id': 'COEDIT_LARGE'}, 'coedit_large', 'COEDIT_LARGE'),
            (
                {'triton_model_name': 'coedit_large_v2'},
                {'rule_id': 'COEDIT_LARGE'},
                'coedit_large_v2',
                'COEDIT_LARGE',
            ),
        ],
    )
    @patch('grammared_language.clients.coedit_client.CoEditClient')
    def test_coedit_identity_defaults_are_independent(
        self,
        mock_coedit,
        serving_overrides,
        grammared_overrides,
        expected_served_name,
        expected_rule_id,
    ):
        create_client_from_config(
            'coedit_large',
            {
                'type': 'coedit',
                'backend': 'triton',
                'serving_config': {
                    'pretrained_model_name_or_path': 'rayliuca/coedit-large-onnx',
                    **serving_overrides,
                },
                'grammared_config': grammared_overrides,
            },
        )

        kwargs = mock_coedit.call_args.kwargs
        assert kwargs['model_name'] == expected_served_name
        assert kwargs['rule_id'] == expected_rule_id

    @patch('grammared_language.clients.gector_client.GectorClient')
    def test_gector_triton_and_rule_defaults_use_logical_name(self, mock_gector):
        create_client_from_config(
            'gector_logical',
            {
                'type': 'gector',
                'backend': 'triton',
                'serving_config': {
                    'pretrained_model_name_or_path': 'publisher/gector-artifact',
                },
            },
        )

        kwargs = mock_gector.call_args.kwargs
        assert kwargs['pretrained_model_name_or_path'] == 'publisher/gector-artifact'
        assert kwargs['triton_model_name'] == 'gector_logical'
        assert kwargs['rule_id'] == 'gector_logical'

    @patch('grammared_language.clients.text2text_base_client.grpcclient')
    def test_coedit_configured_rule_id_reaches_language_tool_matches(self, mock_grpcclient):
        """Smoke test the complete config -> client -> LanguageTool Match.id path."""
        mock_grpcclient.InferenceServerClient = Mock()

        client = create_client_from_config(
            'coedit_large',
            {
                'type': 'coedit',
                'backend': 'triton',
                'serving_config': {
                    'pretrained_model_name_or_path': 'rayliuca/coedit-large-onnx',
                    'triton_model_name': 'coedit_large_smoke',
                },
                'grammared_config': {'rule_id': 'SMOKE_COEDIT_RULE'},
            },
        )

        assert client is not None
        assert client.model_name == 'coedit_large_smoke'
        result = client._pred_postprocess(
            'This are a test.',
            'This is a test.',
        )
        assert result.matches
        assert {match.id for match in result.matches} == {'SMOKE_COEDIT_RULE'}
    
    
    def test_config_requires_serving_config(self):
        """Test that serving_config is required in new format."""
        with pytest.raises(Exception):  # Should raise validation error
            CoEditConfig(
                type='coedit',
                backend='triton'
                # Missing serving_config
            )
