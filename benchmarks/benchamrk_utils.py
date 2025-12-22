
def load_paramas(self,params_path):
    model.load_state_dict(torch.load(params_path, map_location = self.model.device))

def run_test(model,x_test,y_test,test_name, eps):
    model.eval()

    # with torch.no_grad():
    log_file_path = f"results/adv_eval_l2_check/log_RT_{test_name}.txt"

    # adversary = AutoAttack(model, norm='Linf', eps=eps, version='custom', attacks_to_run=['apgd-ce'],log_path=log_file_path)
    adversary = AutoAttack(model, norm='L2', eps=eps, version='custom', attacks_to_run=['apgd-ce'],log_path=log_file_path)
    adversary.apgd.n_restarts = 1
    adversary.run_standard_evaluation(x_test,y_test)
    print("\n --- Test Complete ---")
