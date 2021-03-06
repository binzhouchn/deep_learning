# 注：args, Dataset, transformations和main.py中的一样
tt = pd.read_csv('ds/result.csv', header=None, names=['img'])
test_set = Dataset('ds/test/', img_name_and_label=[(x[0], -1) for x in tt.to_numpy()],
                   transform=transformations['test'])
test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

model = make_model(args)
model.load_state_dict(torch.load('model_2_9187_8833.pth.pth'))  # 加载checkpoints中最好的那个文件比如model_2_9187_8833.pth
if not args.no_cuda:
    model = model.cuda()

preds = []
with torch.no_grad():
    model.eval()
    for i, (input, target) in enumerate(test_loader):
        if not args.no_cuda:
            input = input.cuda()
        output = model(input)
        _, pred = torch.max(output, 1)  # 二分类
        preds += pred

preds = [x.item() for x in preds]
tt['label'] = preds

tt.to_csv('ds/submit.csv', header=None, index=False)



'''
目前线上成绩最好0.864的是单个efficientnet-b5模型跑出，需要加载b5预训练参数（图片resize到456*456）
'''
