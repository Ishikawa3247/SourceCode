# Kaggle-MountainDEW

## gitの使い方
自分の編集をgithubに反映させるやり方
git add => git commit => git push =>マージ

変更内容をlocalに反映させるやり方
git checkout main => git pull

rebaseのやり方
rebaseはコンフリクト(誰かが変更して、自分のファイルには変更が反映されてないとき)の解消に使う。

git add とgit commitまで済ませた状態で、mainにcheckoutしてgit pull origin main

次にgit rebase main

次にgit rebase --cotinue

リベースが完了するので、git push origin <ブランチ名>でマージリクエストを作る。

誰かがmainにpushしてmergeしたときの変更内容をlocalに反映させる方法
各々の作業しているブランチで　git merge main　を行う


