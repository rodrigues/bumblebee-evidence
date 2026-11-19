# Chat template branch coverage

Cases: cases.json, 32 conversations.

| Template | Branches | Reached | Coverage |
|---|---|---|---|
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 10 | 10 | 100% |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 6 | 6 | 100% |
| mistralai/Mistral-7B-Instruct-v0.3 | 52 | 47 | 90% |
| HuggingFaceH4/zephyr-7b-beta | 10 | 10 | 100% |
| unsloth/gemma-7b-it | 10 | 10 | 100% |
| microsoft/Phi-3-mini-4k-instruct | 10 | 10 | 100% |
| microsoft/Phi-3.5-mini-instruct | 10 | 10 | 100% |
| microsoft/Phi-4-mini-instruct | 6 | 6 | 100% |
| Qwen/Qwen3-0.6B | 52 | 50 | 96% |
| Qwen/Qwen3-4B-Instruct-2507 | 34 | 34 | 100% |
| HuggingFaceTB/SmolLM3-3B | 46 | 42 | 91% |
| google/gemma-4-E4B-it | 278 | 228 | 82% |
| NousResearch/Hermes-2-Pro-Llama-3-8B (default) | 4 | 4 | 100% |
| NousResearch/Hermes-2-Pro-Llama-3-8B (tool_use) | 62 | 56 | 90% |
| All | 590 | 523 | 89% |

Untaken branches, by the template line of the tag that branches:

## mistralai/Mistral-7B-Instruct-v0.3

- line 7: `{%- if not tools is defined %}`
- line 13: `{%- set ns = namespace() %}`
- line 18: `{{- raise_exception("After the optional system message, conversation roles must alternate user/assistant/user/`
- line 73: `{%- elif message["role"] == "tool_results" or message["role"] == "tool" %}`
- line 80: `{%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}`

## Qwen/Qwen3-0.6B

- line 20: `{%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.sta`
- line 21: `{%- set ns.multi_step_tool = false %}`

## HuggingFaceTB/SmolLM3-3B

- line 47: `{%- if xml_tools or tools -%}`
- line 52: `{%- for tool in xml_tools[:] -%} {# The slicing makes sure that xml_tools is a list #}`
- line 60: `{%- for tool in python_tools[:] -%} {# The slicing makes sure that python_tools is a list #}`
- line 77: `{%- if reasoning_mode == "/think" -%}`

## google/gemma-4-E4B-it

- line 7: `{%- macro format_parameters(properties, required, filter_keys=false) -%}`
- line 13: `{%- if ns.found_first %},{% endif -%}`
- line 22: `{%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}`
- line 26: `{%- if value['items'] is mapping and value['items'] -%}`
- line 27: `{%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}`
- line 31: `{%- if item_value is not none -%}`
- line 32: `{%- if ns_items.found_first %},{% endif -%}`
- line 36: `{%- if item_value is mapping -%}`
- line 44: `{%- if not loop.last %},{% endif -%}`
- line 47: `{%- elif item_key == 'type' -%}`
- line 48: `{%- if item_value is string -%}`
- line 62: `{%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}`
- line 67: `{%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}`
- line 71: `{%- elif value is mapping -%}`
- line 72: `{%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}`
- line 78: `{%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}`
- line 92: `{%- macro format_function_declaration(tool_data) -%}`
- line 95: `{%- if params -%}`
- line 108: `{%- if params['type'] -%}`
- line 115: `{%- if response_declaration['description'] -%}`
- line 118: `{%- if response_declaration['type'] | upper == 'OBJECT' -%}`
- line 124: `{%- macro format_argument(argument, escape_keys=True) -%}`
- line 135: `{%- if ns.found_first %},{% endif -%}`
- line 137: `{%- if escape_keys -%}`
- line 156: `{%- macro strip_thinking(text) -%}`
- line 159: `{%- if '<|channel>' in part -%}`
- line 160: `{%- set ns.result = ns.result + part.split('<|channel>')[0] -%}`
- line 168: `{%- macro format_tool_response_block(tool_name, response) -%}`
- line 193: `{%- if enable_thinking -%}`
- line 200: `{%- elif messages[0]['content'] is sequence -%}`
- line 210: `{{- format_function_declaration(tool) | trim -}}`
- line 221: `{%- if loop_messages[i]['role'] == 'user' -%}`
- line 228: `{%- if message['role'] != 'tool' -%}`
- line 240: `{%- if thinking_text and thinking_gate -%}`
- line 251: `{%- if ns_args.found_first %},{% endif -%}`
- line 257: `{{- raise_exception(`
- line 272: `{{- format_tool_response_block(tool_response['name'] | default('unknown', true), tool_response['response']) -}`
- line 273: `{%- set ns_tr_out.flag = true -%}`
- line 281: `{%- elif loop_messages[k]['role'] != 'tool' -%}`
- line 288: `{%- if tc.get('id') == follow.get('tool_call_id') -%}`
- line 296: `{%- elif tool_body is sequence and tool_body is not string -%}`
- line 299: `{%- if part.get('type') == 'text' -%}`
- line 314: `{{- format_tool_response_block(ns_tname.name, tool_body) -}}`
- line 316: `{%- set ns_tr_out.flag = true -%}`
- line 341: `{%- elif item.get('type') == 'video' -%}`
- line 355: `{%- if loop_messages[j]['role'] != 'tool' -%}`
- line 356: `{%- set next_nt.role = loop_messages[j]['role'] -%}`
- line 371: `{%- elif not (ns_tr_out.flag and not has_content and not next_nt.found) -%}`

## NousResearch/Hermes-2-Pro-Llama-3-8B (tool_use)

- line 1: `{%- macro json_to_python_type(json_spec) %}`
- line 19: `{%- elif json_spec.type is iterable %}`
- line 23: `{%- if not loop.last %}`
- line 39: `{%- if tool.function is defined %}`
- line 112: `{%- if tool_call.arguments is defined %}`
- line 126: `{%- elif message.role == "tool" %}`
