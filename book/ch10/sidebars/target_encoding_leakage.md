# Sidebar: target encoding and the second shape of label leakage

Binary indicators are not the only way to encode a high-cardinality
categorical feature. A common alternative, particularly in Kaggle
competitions and any tabular-ML setting, is **target encoding**:
replace each category with the mean of the target variable for
that category.

For our setup, that would mean: for each individual skill (Python,
SQL, etc.), compute the fraction of postings with that skill that
are AI Engineer. Then encode the "skills" feature for each posting
as the mean of these fractions across its skills.

This often works. It compresses a 200-skill one-hot encoding into
a single numeric column. The classifier can learn from it directly.

It also leaks the target into the features in a subtle way, and
this is the second shape of the leakage problem Chapter 9 named.
The first shape was training a classifier on the same fields the
labels were derived from. The second shape is computing the
encoding on the full dataset before splitting train and test:

    # WRONG — leaks test-set labels into training features
    skill_to_target_mean = df.groupby("skill")["role_category"].mean()
    df["skill_encoding"] = df["skills"].apply(
        lambda skills: skill_to_target_mean[skills].mean()
    )
    X_train, X_test = train_test_split(df)

The classifier sees a feature whose value at training time was
influenced by the test-set rows' labels. F1 looks 3–8% better than
it should. Production performance is worse than the test set
suggested it would be.

The fix is **out-of-fold target encoding**: compute the encoding
on each cross-validation fold using only the other folds' data:

    # RIGHT — fold-aware target encoding
    from sklearn.model_selection import KFold
    encoding = pd.Series(index=df.index, dtype=float)
    for train_idx, val_idx in KFold(5).split(df):
        train = df.iloc[train_idx]
        skill_means = train.groupby("skill")["role_category"].mean()
        encoding.iloc[val_idx] = df.iloc[val_idx]["skills"].apply(
            lambda s: skill_means[s].mean()
        )
    df["skill_encoding"] = encoding

For Chapter 10, we don't use target encoding at all. With only
ten high-signal skills, binary indicators are simpler, do not
leak, and produce comparable F1. But if you take this codebase to
a larger dataset with thousands of distinct skills, target
encoding becomes attractive, and at that point this sidebar's
fold-aware pattern is what you copy.

The shape of the lesson is the same as Chapter 9's: **leakage
shows up wherever any function of the target influences any
feature, even one transformation removed**. Once you start looking
for it, you find it in places you didn't expect: target encoding,
fitting a scaler on all data before split, fitting a PCA on all
data before split, computing class weights before split. The
pattern generalises. Most subtle ML bugs are this pattern in
disguise.
