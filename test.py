from server import train


def test_training():
    regr, X_test, y_test = train(["complexAge", "totalRooms", "totalBedrooms", "complexInhabitants", "apartmentsNr", "medianCompexValue"])

    assert regr.score(X_test, y_test) >= 0.21
    assert False
