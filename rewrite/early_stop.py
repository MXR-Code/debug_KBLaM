class EarlyStopping():
    def __init__(self, patience=50):
        self.val_loss_list = list()
        self.best_model_parameter_state_dict = None
        self.total_patience = patience
        self.remain_patience = self.total_patience
        self.is_stop = False
        self.min_val_loss = None

    def record(self, model, now_val_loss):
        if self.min_val_loss is None:
            self.min_val_loss = now_val_loss

        if now_val_loss < self.min_val_loss:
            self.best_model_parameter_state_dict = model.state_dict()
            self.remain_patience = self.total_patience
            self.min_val_loss = now_val_loss

        if now_val_loss >= self.min_val_loss:
            self.remain_patience = self.remain_patience - 1

        if self.remain_patience <= 0:
            self.is_stop = True

        print('patience:', self.remain_patience)

