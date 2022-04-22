/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <fcntl.h>

#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#include "paddle/fluid/pybind/fleet_py.h"

#include <map>
#include <memory>
#include <string>
#include <vector>
// #include <cuda_runtime.h>

#include "paddle/fluid/distributed/index_dataset/index_sampler.h"
#include "paddle/fluid/distributed/index_dataset/index_wrapper.h"
#include "paddle/fluid/distributed/ps/service/communicator/communicator.h"
#include "paddle/fluid/distributed/ps/service/communicator/communicator_common.h"
#include "paddle/fluid/distributed/ps/service/env.h"
#include "paddle/fluid/distributed/ps/service/graph_brpc_client.h"
#include "paddle/fluid/distributed/ps/service/heter_client.h"
#include "paddle/fluid/distributed/ps/service/ps_service/graph_py_service.h"
#include "paddle/fluid/distributed/ps/wrapper/fleet.h"

#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_ps_table.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
#include "paddle/fluid/framework/fleet/heter_ps/optimizer.cuh.h"
#include "paddle/fluid/platform/cuda_device_guard.h"

namespace py = pybind11;
using paddle::distributed::CommContext;
using paddle::distributed::Communicator;
using paddle::distributed::FleetWrapper;
using paddle::distributed::HeterClient;
using paddle::distributed::GraphPyService;
using paddle::distributed::GraphNode;
using paddle::distributed::GraphPyServer;
using paddle::distributed::GraphPyClient;
using paddle::distributed::FeatureNode;
// using paddle::framework::GpuPsGraphTable;
// using paddle::framework::HeterPsResource;

namespace paddle {
namespace pybind {
void BindDistFleetWrapper(py::module* m) {
  py::class_<FleetWrapper, std::shared_ptr<FleetWrapper>>(*m,
                                                          "DistFleetWrapper")
      .def(py::init([]() { return FleetWrapper::GetInstance(); }))
      .def("load_sparse", &FleetWrapper::LoadSparseOnServer)
      .def("load_model", &FleetWrapper::LoadModel)
      .def("load_one_table", &FleetWrapper::LoadModelOneTable)
      .def("init_server", &FleetWrapper::InitServer)
      .def("run_server",
           (uint64_t (FleetWrapper::*)(void)) & FleetWrapper::RunServer)
      .def("run_server", (uint64_t (FleetWrapper::*)(          // NOLINT
                             const std::string&, uint32_t)) &  // NOLINT
                             FleetWrapper::RunServer)
      .def("init_worker", &FleetWrapper::InitWorker)
      .def("push_dense_params", &FleetWrapper::PushDenseParamSync)
      .def("pull_dense_params", &FleetWrapper::PullDenseVarsSync)
      .def("save_all_model", &FleetWrapper::SaveModel)
      .def("save_one_model", &FleetWrapper::SaveModelOneTable)
      .def("recv_and_save_model", &FleetWrapper::RecvAndSaveTable)
      .def("sparse_table_stat", &FleetWrapper::PrintTableStat)
      .def("stop_server", &FleetWrapper::StopServer)
      .def("stop_worker", &FleetWrapper::FinalizeWorker)
      .def("barrier", &FleetWrapper::BarrierWithTable)
      .def("shrink_sparse_table", &FleetWrapper::ShrinkSparseTable)
      .def("set_clients", &FleetWrapper::SetClients)
      .def("get_client_info", &FleetWrapper::GetClientsInfo)
      .def("create_client2client_connection",
           &FleetWrapper::CreateClient2ClientConnection);
}

void BindPSHost(py::module* m) {
  py::class_<distributed::PSHost>(*m, "PSHost")
      .def(py::init<const std::string&, uint32_t, uint32_t>())
      .def("serialize_to_string", &distributed::PSHost::SerializeToString)
      .def("parse_from_string", &distributed::PSHost::ParseFromString)
      .def("to_uint64", &distributed::PSHost::SerializeToUint64)
      .def("from_uint64", &distributed::PSHost::ParseFromUint64)
      .def("to_string", &distributed::PSHost::ToString);
}

void BindCommunicatorContext(py::module* m) {
  py::class_<CommContext>(*m, "CommContext")
      .def(
          py::init<const std::string&, const std::vector<std::string>&,
                   const std::vector<std::string>&, const std::vector<int64_t>&,
                   const std::vector<std::string>&, int, bool, bool, bool, int,
                   bool, bool, int64_t>())
      .def("var_name", [](const CommContext& self) { return self.var_name; })
      .def("trainer_id",
           [](const CommContext& self) { return self.trainer_id; })
      .def("table_id", [](const CommContext& self) { return self.table_id; })
      .def("program_id",
           [](const CommContext& self) { return self.program_id; })
      .def("split_varnames",
           [](const CommContext& self) { return self.splited_varnames; })
      .def("split_endpoints",
           [](const CommContext& self) { return self.epmap; })
      .def("sections",
           [](const CommContext& self) { return self.height_sections; })
      .def("aggregate", [](const CommContext& self) { return self.merge_add; })
      .def("is_sparse", [](const CommContext& self) { return self.is_sparse; })
      .def("is_distributed",
           [](const CommContext& self) { return self.is_distributed; })
      .def("origin_varnames",
           [](const CommContext& self) { return self.origin_varnames; })
      .def("is_tensor_table",
           [](const CommContext& self) { return self.is_tensor_table; })
      .def("is_datanorm_table",
           [](const CommContext& self) { return self.is_datanorm_table; })
      .def("__str__", [](const CommContext& self) { return self.print(); });
}

using paddle::distributed::AsyncCommunicator;
using paddle::distributed::GeoCommunicator;
using paddle::distributed::RecvCtxMap;
using paddle::distributed::RpcCtxMap;
using paddle::distributed::SyncCommunicator;
using paddle::framework::Scope;

void BindDistCommunicator(py::module* m) {
  // Communicator is already used by nccl, change to DistCommunicator
  py::class_<Communicator, std::shared_ptr<Communicator>>(*m,
                                                          "DistCommunicator")
      .def(py::init([](const std::string& mode, const std::string& dist_desc,
                       const std::vector<std::string>& host_sign_list,
                       const RpcCtxMap& send_ctx, const RecvCtxMap& recv_ctx,
                       Scope* param_scope,
                       std::map<std::string, std::string>& envs) {
        if (mode == "ASYNC") {
          Communicator::InitInstance<AsyncCommunicator>(
              send_ctx, recv_ctx, dist_desc, host_sign_list, param_scope, envs);
        } else if (mode == "SYNC") {
          Communicator::InitInstance<SyncCommunicator>(
              send_ctx, recv_ctx, dist_desc, host_sign_list, param_scope, envs);
        } else if (mode == "GEO") {
          Communicator::InitInstance<GeoCommunicator>(
              send_ctx, recv_ctx, dist_desc, host_sign_list, param_scope, envs);
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "unsuported communicator MODE"));
        }
        return Communicator::GetInstantcePtr();
      }))
      .def("stop", &Communicator::Stop)
      .def("start", &Communicator::Start)
      .def("push_sparse_param", &Communicator::RpcSendSparseParam)
      .def("is_running", &Communicator::IsRunning)
      .def("init_params", &Communicator::InitParams)
      .def("pull_dense", &Communicator::PullDense)
      .def("create_client_to_client_connection",
           &Communicator::CreateC2CConnection)
      .def("get_client_info", &Communicator::GetClientInfo)
      .def("set_clients", &Communicator::SetClients);
}

void BindHeterClient(py::module* m) {
  py::class_<HeterClient, std::shared_ptr<HeterClient>>(*m, "HeterClient")
      .def(py::init([](const std::vector<std::string>& endpoints,
                       const std::vector<std::string>& previous_endpoints,
                       const int& trainer_id) {
        return HeterClient::GetInstance(endpoints, previous_endpoints,
                                        trainer_id);
      }))
      .def("stop", &HeterClient::Stop);
}

void BindGraphNode(py::module* m) {
  py::class_<GraphNode>(*m, "GraphNode")
      .def(py::init<>())
      .def("get_id", &GraphNode::get_id)
      .def("get_feature", &GraphNode::get_feature);
}
void BindGraphPyFeatureNode(py::module* m) {
  py::class_<FeatureNode>(*m, "FeatureNode")
      .def(py::init<>())
      .def("get_id", &GraphNode::get_id)
      .def("get_feature", &GraphNode::get_feature);
}

void BindGraphPyService(py::module* m) {
  py::class_<GraphPyService>(*m, "GraphPyService").def(py::init<>());
}

void BindGraphPyServer(py::module* m) {
  py::class_<GraphPyServer>(*m, "GraphPyServer")
      .def(py::init<>())
      .def("start_server", &GraphPyServer::start_server)
      .def("set_up", &GraphPyServer::set_up)
      .def("add_table_feat_conf", &GraphPyServer::add_table_feat_conf);
}
void BindGraphPyClient(py::module* m) {
  py::class_<GraphPyClient>(*m, "GraphPyClient")
      .def(py::init<>())
      .def("load_edge_file", &GraphPyClient::load_edge_file)
      .def("load_node_file", &GraphPyClient::load_node_file)
      .def("set_up", &GraphPyClient::set_up)
      .def("add_table_feat_conf", &GraphPyClient::add_table_feat_conf)
      .def("pull_graph_list", &GraphPyClient::pull_graph_list)
      .def("start_client", &GraphPyClient::start_client)
      .def("batch_sample_neighboors", &GraphPyClient::batch_sample_neighbors)
      .def("batch_sample_neighbors", &GraphPyClient::batch_sample_neighbors)
      .def("use_neighbors_sample_cache",
           &GraphPyClient::use_neighbors_sample_cache)
      .def("remove_graph_node", &GraphPyClient::remove_graph_node)
      .def("random_sample_nodes", &GraphPyClient::random_sample_nodes)
      .def("stop_server", &GraphPyClient::StopServer)
      .def("get_node_feat",
           [](GraphPyClient& self, std::string node_type,
              std::vector<int64_t> node_ids,
              std::vector<std::string> feature_names) {
             auto feats =
                 self.get_node_feat(node_type, node_ids, feature_names);
             std::vector<std::vector<py::bytes>> bytes_feats(feats.size());
             for (int i = 0; i < feats.size(); ++i) {
               for (int j = 0; j < feats[i].size(); ++j) {
                 bytes_feats[i].push_back(py::bytes(feats[i][j]));
               }
             }
             return bytes_feats;
           })
      .def("set_node_feat",
           [](GraphPyClient& self, std::string node_type,
              std::vector<int64_t> node_ids,
              std::vector<std::string> feature_names,
              std::vector<std::vector<py::bytes>> bytes_feats) {
             std::vector<std::vector<std::string>> feats(bytes_feats.size());
             for (int i = 0; i < bytes_feats.size(); ++i) {
               for (int j = 0; j < bytes_feats[i].size(); ++j) {
                 feats[i].push_back(std::string(bytes_feats[i][j]));
               }
             }
             self.set_node_feat(node_type, node_ids, feature_names, feats);
             return;
           })
      .def("bind_local_server", &GraphPyClient::bind_local_server);
}

// void BindGPUGraphTable(py::module* m) {
//   py::class_<GpuPsGraphTable>(*m, "GpuPsGraphTable")
//       .def("__init__", [](GpuPsGraphTable &instance, std::vector<int> device_id_mapping2){
//           // int gpu_num = device_id_mapping.size();
//
//           std::vector<int> device_id_mapping;
//           for (int i = 0; i < 2; i++) device_id_mapping.push_back(i);
//           std::shared_ptr<HeterPsResource> resource =
//               std::make_shared<HeterPsResource>(device_id_mapping);
//           resource->enable_p2p();
//           int use_nv = 1;
//           new (&instance) GpuPsGraphTable(resource, use_nv);
//         })
//       .def("init_cpu_table",
//               [](GpuPsGraphTable& self) {
//                   ::paddle::distributed::GraphParameter table_proto;
//                   table_proto.set_shard_num(24);
//                   self.init_cpu_table(table_proto);
//         });
      // .def("load",
      //         [](GpuPsGraphTable& self) {
      //             std::vector<paddle::framework::GpuPsCommGraph> vec;
      //             int n = 10;
      //             std::vector<int64_t> ids0, ids1;
      //             for (int i = 0; i < n; i++) {
      //               self.cpu_graph_table->add_comm_edge(i, (i + 1) % n);
      //               self.cpu_graph_table->add_comm_edge(i, (i - 1 + n) % n);
      //               if (i % 2 == 0) ids0.push_back(i);
      //             }
      //             self.cpu_graph_table->build_sampler();
      //             ids1.push_back(5);
      //             vec.push_back(self.cpu_graph_table->make_gpu_ps_graph(ids0));
      //             vec.push_back(self.cpu_graph_table->make_gpu_ps_graph(ids1));
      //             vec[0].display_on_cpu();
      //             vec[1].display_on_cpu();
      //             self.build_graph_from_cpu(vec);
      //         })
      // // .def("sample_neighboors", &GraphPyClient::batch_sample_neighbors)
      // .def("sample_neighboors", [](GpuPsGraphTable& self, std::vector<int> nodes) {
      //             int64_t cpu_key[3] = {0, 1, 2};
      //             void *key;
      //             platform::CUDADeviceGuard guard(0);
      //             cudaMalloc((void **)&key, 3 * sizeof(int64_t));
      //             cudaMemcpy(key, cpu_key, 3 * sizeof(int64_t), cudaMemcpyHostToDevice);
      //             // auto neighbor_sample_res = g.graph_neighbor_sample(0, (int64_t *)key, 2,
      //             // 3);
      //             auto neighbor_sample_res =
      //                 self.graph_neighbor_sample_v2(0, (int64_t *)key, 2, 3, true);
      //             int64_t *res = new int64_t[7];
      //             cudaMemcpy(res, neighbor_sample_res->val, 3 * 2 * sizeof(int64_t),
      //                        cudaMemcpyDeviceToHost);
      //             int *actual_sample_size = new int[3];
      //             cudaMemcpy(actual_sample_size, neighbor_sample_res->actual_sample_size,
      //                        3 * sizeof(int),
      //                        cudaMemcpyDeviceToHost);  // 3, 1, 3
      //
      //             return actual_sample_size;
      //         });
// }


using paddle::distributed::TreeIndex;
using paddle::distributed::IndexWrapper;
using paddle::distributed::IndexNode;

void BindIndexNode(py::module* m) {
  py::class_<IndexNode>(*m, "IndexNode")
      .def(py::init<>())
      .def("id", [](IndexNode& self) { return self.id(); })
      .def("is_leaf", [](IndexNode& self) { return self.is_leaf(); })
      .def("probability", [](IndexNode& self) { return self.probability(); });
}

void BindTreeIndex(py::module* m) {
  py::class_<TreeIndex, std::shared_ptr<TreeIndex>>(*m, "TreeIndex")
      .def(py::init([](const std::string name, const std::string path) {
        auto index_wrapper = IndexWrapper::GetInstancePtr();
        index_wrapper->insert_tree_index(name, path);
        return index_wrapper->get_tree_index(name);
      }))
      .def("height", [](TreeIndex& self) { return self.Height(); })
      .def("branch", [](TreeIndex& self) { return self.Branch(); })
      .def("total_node_nums",
           [](TreeIndex& self) { return self.TotalNodeNums(); })
      .def("emb_size", [](TreeIndex& self) { return self.EmbSize(); })
      .def("get_all_leafs", [](TreeIndex& self) { return self.GetAllLeafs(); })
      .def("get_nodes",
           [](TreeIndex& self, const std::vector<uint64_t>& codes) {
             return self.GetNodes(codes);
           })
      .def("get_layer_codes",
           [](TreeIndex& self, int level) { return self.GetLayerCodes(level); })
      .def("get_ancestor_codes",
           [](TreeIndex& self, const std::vector<uint64_t>& ids, int level) {
             return self.GetAncestorCodes(ids, level);
           })
      .def("get_children_codes",
           [](TreeIndex& self, uint64_t ancestor, int level) {
             return self.GetChildrenCodes(ancestor, level);
           })
      .def("get_travel_codes",
           [](TreeIndex& self, uint64_t id, int start_level) {
             return self.GetTravelCodes(id, start_level);
           });
}

void BindIndexWrapper(py::module* m) {
  py::class_<IndexWrapper, std::shared_ptr<IndexWrapper>>(*m, "IndexWrapper")
      .def(py::init([]() { return IndexWrapper::GetInstancePtr(); }))
      .def("insert_tree_index", &IndexWrapper::insert_tree_index)
      .def("get_tree_index", &IndexWrapper::get_tree_index)
      .def("clear_tree", &IndexWrapper::clear_tree);
}

using paddle::distributed::IndexSampler;
using paddle::distributed::LayerWiseSampler;

void BindIndexSampler(py::module* m) {
  py::class_<IndexSampler, std::shared_ptr<IndexSampler>>(*m, "IndexSampler")
      .def(py::init([](const std::string& mode, const std::string& name) {
        if (mode == "by_layerwise") {
          return IndexSampler::Init<LayerWiseSampler>(name);
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Unsupported IndexSampler Type!"));
        }
      }))
      .def("init_layerwise_conf", &IndexSampler::init_layerwise_conf)
      .def("init_beamsearch_conf", &IndexSampler::init_beamsearch_conf)
      .def("sample", &IndexSampler::sample);
}
}  // end namespace pybind
}  // namespace paddle
