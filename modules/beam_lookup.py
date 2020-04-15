import numpy as np


def get_dates():
    # Must update get_dates() and get_beam_stats() together.
    dates = np.array(['190628', '190722', '190821', '190905', '190916', '191002', '191008', '191023', '191120'])

    # *** Until we have a full beam complement ***:
    dates = np.array(['190628', '191002', '191023'])
    return dates


def get_beam_stats(dates):
    # get_beam_stats() assumes all beams of a date are good unless told otherwise!
    beams = np.ones([len(dates), 40])
    beams[dates == '190628', :32] = 0
    beams[dates == '190722', :] = 0
    beams[dates == '190821', 4] = 0
    beams[dates == '190905', 18:22] = 0
    beams[dates == '190905', 27:34] = 0
    beams[dates == '190916', 27:] = 0
    # Big change in beam quality after 1 October 2019 (see in difference images).
    beams[dates == '191002', 8:15] = 1   # best we have at the moment, but would flip to 0 if better comes!
    beams[dates == '191002', 4:6] = 1   # best we have at the moment, but would flip to 0 if better comes!
    beams[dates == '191008', 1:16] = 0
    beams[dates == '191008', 18:22] = 0
    beams[dates == '191008', 26] = 0
    beams[dates == '191023', :] = beams[dates == '191002', :]
    beams[dates == '191120', :] = 0

    # *** Until we have a full beam complement ***:
    beams = np.ones([len(dates), 40])

    return beams


def nearest_date(dates, taskid):
    distance = [int(d)-int(taskid) for d in dates]
    index = np.where(np.min(np.abs(distance)) == np.abs(distance))
    if len(index[0]) > 1:
        index = index[0][0]  # equivalent to floor if there are two equidistant choices.
    return dates[index][0]


def floor_date(dates, taskid):
    # This is untested, but should work I think??
    distance = [int(d)-int(taskid) for d in dates]
    floor = distance[distance <= 0]
    index = np.where(np.max(floor) == floor)
    return dates[index][0]


def model_lookup(taskid, beam):
    # Assumes running on happili-05:
    model_dir = '/tank/apertif/driftscans/fits_files/'
    all_dates = get_dates()
    all_beam_stats = get_beam_stats(all_dates)
    if beam > all_beam_stats.shape[1] - 1:
        print("{}: Pick a valid beam number 0-39.".format(beam))
        exit()
    beam_stats = all_beam_stats[:, beam]

    # Divide into before & after beam attenuation on October 1st (big impact on beam quality)!
    taskid = str(taskid)[:6]
    if int(taskid) < 191001:
        # *** Until we have a full beam complement ***:
        index = np.where(all_dates == '190628')[0][0]
        # index = np.where(all_dates == '190916')[0][0]
        dates = all_dates[:index + 1]
        beams = beam_stats[:index + 1]
    else:
        index = np.where(all_dates == '191002')[0][0]
        dates = all_dates[index:]
        beams = beam_stats[index:]

    if np.all(beams == 0):
        print("No good beam model options for period when this was observed. Do more drift scans (or edit code).")
        exit()
    elif len(beams[beams == 1]) == 1:
        # If only one good beam model exists, use it.
        best = dates[beams == 1][0]
    else:
        # Use nearest.  Don't have enough beam statistics for floor, I think.
        dates = dates[beams == 1]
        best = nearest_date(dates, taskid)

    # *** Until we have a full beam complement ***:
    if beam >= 32:
        best = '191002'

    model = model_dir + '{}/beam_models/chann_9/{}_{:02}_I_model.fits'.format(best, best, beam)

    return model


# Test code:
# for b in range(40):
#     try:
#         fitsfile = model_lookup(190916, b)
#     except:
#         print(b)
#         continue
#     print(b, fitsfile)
