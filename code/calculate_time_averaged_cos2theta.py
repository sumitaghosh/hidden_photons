from numpy import zeros, sort, pi, arccos, sin, cos, random, trapz, array, append, argmax, argmin, linspace, logspace, \
    add, subtract, divide, sqrt, load, savez, abs, seterr, transpose
from matplotlib.pyplot import figure, plot, legend, show, xlabel, ylabel, imshow, colorbar, title, vlines
from matplotlib.colors import LogNorm
from scipy.integrate import cumtrapz
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from time import time

seterr(divide='raise')  # so that try-except can be used later to filter out warnings
debugging = True  # for when the code isn't working, and it would be helpful to have more print statements running
start_time = time()
print('code started running at', start_time)


def print_preamble():
    return str(time() - start_time) + '    '


def costh2Z(time_days, costh_X, phi_X, experiment_latitude):
    """
    Taken from https://github.com/cajohare/DarkPhotonCookbook/blob/main/code/LabFuncs.py
    :param time_days: measurement time in days
    :param costh_X: cosine of the axial angle of the hidden photon polarization
    :param phi_X: azimuthal angle of the hidden photon polarization
    :param experiment_latitude: latitude of the experiment
    :return: square of the dot product between the axis of the haloscope and the polarization of the hidden photon,
    assuming the haloscope points towards the zenith
    """
    experiment_latitude *= pi / 180
    wt = 2 * pi * time_days
    th_X = arccos(costh_X)
    sthx = sin(th_X)
    clat = cos(experiment_latitude)
    return (sthx * cos(phi_X) * clat * cos(wt) + sthx * sin(phi_X) * clat * sin(wt)
            + costh_X * sin(experiment_latitude)) ** 2


def get_time_averaged_cos2th_pdf_function(time_days, latitude, nt=2000, ngen=100000, make_plots=False, block=True):
    """
    Calculates the CDF of cos2theta, the square of the dot product between the polarization of the hidden photon and the
    axis of the haloscope cavity, for a given desired CL. Code is modified from
    https://github.com/cajohare/DarkPhotonCookbook/blob/main/code/Polarisation_angles.ipynb.
    :param time_days: How long the measurement took, in days
    :param latitude: latitude in degrees
    :param nt: integer number of values of time to plot
    :param ngen: integer number of DP polarizations to sample over
    :param make_plots: boolean option to make plots
    :param block: if plots are made, boolean option to require all plots be closed before the next bit of code runs
    :return: 
    """
    # uniform distribution over the unit sphere for the polarization
    costh_X = 2 * random.uniform(size=ngen) - 1
    phi_X = 2 * pi * random.uniform(size=ngen)
    T = linspace(0, time_days, nt + 1)  # time array in days
    cos2theta_calculated_values = zeros(shape=(ngen, nt + 1))  # will be populated by cos^2(theta) values

    for i in range(0, nt + 1):
        curr_T = T[i]
        cos2theta_calculated_values[:, i] = costh2Z(curr_T, costh_X, phi_X, latitude)

    # Calculate the array of time averages
    cos2theta_time_av = trapz(cos2theta_calculated_values, x=T) / T[-1]
    cdf_x_axis_values = sort(cos2theta_time_av)
    cdf_y_axis_values = array(range(ngen)) / ngen
    smoothed_cdf_xval = savgol_filter(cdf_x_axis_values, 199, 1)

    if make_plots:
        # plot the cdf we've just constructed
        figure()
        plot(append(0, append(cdf_x_axis_values, 1)), append(0, append(cdf_y_axis_values, 1)),
             label='%.2f day measurement' % time_days)
        plot(append(0, append(smoothed_cdf_xval, 1)), append(0, append(cdf_y_axis_values, 1)),
             label='smoothed %.2f day measurement' % time_days)

        # for kicks, compare to the CDF of a one-hour measurement
        stop_index = argmin(abs(T - 1 / 24)) + 1  # the measurement only lasts 1 hour
        c_a0 = trapz(cos2theta_calculated_values[:, :stop_index], x=T[:stop_index]) / T[stop_index - 1]
        plot(sort(c_a0), cdf_y_axis_values, label='1-hr measurement')

        plot([0, 1], [0.05, 0.05], 'k--')
        plot([0, 1], [0.10, 0.10], 'k--')
        xlabel('values of $\\cos^2\\theta$')
        ylabel('CDF of $\\cos^2\\theta$')
        legend()
        show(block=False)

    # take the derivative to get the pdf
    skip = 1

    pdf_y_axis_values = divide(subtract(cdf_y_axis_values[skip:], cdf_y_axis_values[:-skip]),
                               subtract(smoothed_cdf_xval[skip:], smoothed_cdf_xval[:-skip]))
    pdf_x_axis_values = add(smoothed_cdf_xval[skip:], smoothed_cdf_xval[:-skip]) / 2.

    if make_plots:
        figure()
        plot(pdf_x_axis_values, pdf_y_axis_values, label='derivative of smoothed')

    split_index = argmax(pdf_y_axis_values)
    smooth_over = min(599, split_index // 2)
    left_pdf = pdf_y_axis_values[:split_index + 1]
    left_cos = pdf_x_axis_values[:split_index + 1]
    left_cos_fit, left_pdf_fit = savgol_filter((left_cos, left_pdf), smooth_over, 1)
    right_pdf = pdf_y_axis_values[split_index:]
    right_cos = pdf_x_axis_values[split_index:]
    right_cos_fit, right_pdf_fit = savgol_filter((right_cos, right_pdf), smooth_over, 1)
    pdf_x_axis_values = append(left_cos_fit, right_cos_fit)
    pdf_y_axis_values = append(left_pdf_fit, right_pdf_fit)
    if make_plots:
        plot(pdf_x_axis_values, pdf_y_axis_values, label='smoothed derivative of smoothed')
        print(print_preamble(), 'integral of pdf of cos2theta is', trapz(pdf_y_axis_values, x=pdf_x_axis_values))

        xlabel('values of $\\cos^2\\theta$')
        ylabel('PDF of $\\cos^2\\theta$')
        legend()
        show(block=block)

    pdf_y_axis_values[0] = 0
    pdf_y_axis_values[-1] = 0
    pdf_x_axis_values = append(0, append(pdf_x_axis_values, 1))
    pdf_y_axis_values = append(0, append(pdf_y_axis_values, 0))
    savez('CAPP_pdf_to_interpolate.npz', pdf_x_axis_values=pdf_x_axis_values, pdf_y_axis_values=pdf_y_axis_values)
    pdf_function = interp1d(pdf_x_axis_values, pdf_y_axis_values)
    return pdf_function


def get_instantaneous_cos2theta_pdf(cos2theta):
    """
    The PDF of the square of a uniform random variable X that goes from [0, 1] is equal to 1 / 2 * X.
    We know cos(theta) is a uniform random variable because for a uniform distribution over a unit sphere, the z-value
    of any given point is also uniformly distributed over [-1,1], and we can align the z-axis with the cavity axis.
    :param cos2theta: Square of the uniform random variable {cos(theta) = dot product of polarization and cavity axis}.
    :return: the PDF of cos2theta for a given value of cos2theta
    """
    return 1 / (2 * sqrt(cos2theta))


def get_cos2theta_and_pdf_array(cos2theta_pdf_function, num_pts=500, make_plot=False, block=True):
    """
    Makes sure that the function we use for the PDF of cos2theta integrates to 1 - it does analytically, but since we're
    integrating numerically, we can't start at 0. This means we have to pick some teeny tiny small value instead.
    Because the PDF changes rapidly at infinitesimal values of cos2theta but then changes more slowly, we want to use
    logspace() for the values of cos2theta
    :param cos2theta_pdf_function: which function to use for the PDF of cos2theta, dependent on timing information
    :param num_pts: Number of points for the simulation. You can input a lower number if your computer is slow.
    :param make_plot: boolean on whether to make a plot of the cos2theta PDF
    :param block: if plots are made, boolean option to require all plots be closed before the next bit of code runs
    :return: an array of values for cos2theta that will allow the PDF to integrate to 1
    """
    try:
        cos2theta_array = linspace(0, 1, num_pts)
        cos2theta_pdf = cos2theta_pdf_function(cos2theta_array)
        hopefully_equals_1 = trapz(cos2theta_pdf, cos2theta_array)
        if debugging:
            print(print_preamble(), 'current integral =', hopefully_equals_1, 'when using linspace')
    except FloatingPointError:
        starting_exponent = -9  # hard coding this in from trial and error
        cos2theta_array = logspace(starting_exponent, 0, num_pts)
        cos2theta_pdf = cos2theta_pdf_function(cos2theta_array)
        hopefully_equals_1 = trapz(cos2theta_pdf, cos2theta_array)
        if debugging:
            print(print_preamble(), 'current integral =', hopefully_equals_1, 'when starting exponent =',
                  starting_exponent)
        while hopefully_equals_1 > 1.01 or hopefully_equals_1 < 0.99:
            if hopefully_equals_1 > 1.01:
                starting_exponent += 0.1
            else:
                starting_exponent -= 0.1
            cos2theta_array = logspace(starting_exponent, 0, num_pts)
            cos2theta_pdf = cos2theta_pdf_function(cos2theta_array)
            hopefully_equals_1 = trapz(cos2theta_pdf, cos2theta_array)
            if debugging:
                print(print_preamble(), 'current integral =', hopefully_equals_1, 'when starting exponent =',
                      starting_exponent)
    print(print_preamble(), 'The PDF of cos2theta over the range %f to %f integrates to %f' % (
        cos2theta_array[0], cos2theta_array[-1],
        hopefully_equals_1))
    if make_plot:
        figure()
        plot(cos2theta_array, cos2theta_pdf)
        xlabel('$<\\cos^2(\\theta)>$')
        ylabel('PDF of $<\\cos^2(\\theta)>$')
        show(block=block)
    return cos2theta_array, cos2theta_pdf


def get_joint_pdf(x, cos2theta, cos2theta_pdf, x0=1.645 / 0.024):
    """
    The joint PDF is found by multiplying the individual PDFs of the two random variables together. Our two random
    variables are:
    1) the normalized power, which is a gaussian random variable with a value of 0 and a standard deviation of 1, and
       its mean is suppressed by cos2theta
    2) cos2theta, which is the square of the dot product between the polarization of the hidden photon and the cavity's
       axis. The dot product is a uniform random variable, so cos2theta is the square of a uniform random variable.
    :param x: The normalized power
    :param cos2theta: the square of the dot product between the hidden photon polarization and the cavity axis
    :param cos2theta_pdf: PDF of cos2theta at specific value of cos2theta
    :param x0: the true mean of the power fluctuation, which is suppressed by cos2theta
    :return: the joint PDF between the normalized power and cos2theta
    """
    normalized_power_pdf = norm.pdf(x, x0 * cos2theta, 1)  # this is the PDF of a normalized gaussian random variable
    return cos2theta_pdf * normalized_power_pdf


def find_cos2theta_from_joint_pdf(cos2theta_pdf_function, CL_percentage, start_test_value=0.02, end_test_value=0.1,
                                  num_pts=500, num_test_values=10, make_plots=True, block=True):
    """
    Calculates cos2theta for the desired % CL using interpolation
    :param cos2theta_pdf_function: which function to use for the PDF of cos2theta, dependent on timing information
    :param CL_percentage: confidence limit desired for the hidden photon exclusion out of 100
    :param start_test_value: lowest value of cos2theta to test (will be used for interpolation)
    :param end_test_value: highest value of cos2theta to test (will be used for interpolation)
    :param num_pts: Number of points for the simulation. You can input a lower number if your computer is slow.
    :param num_test_values: Number of test values for cos2theta
    :param make_plots: boolean on whether to make a plot of the cos2theta PDF
    :param block: if plots are made, boolean option to require all plots be closed before the next bit of code runs
    :return: calculated cos2theta value
    """
    x_array = linspace(-5, 0, num_pts)  # possible values of the normalized power fluctuation
    c_array, cos2theta_pdf_array = get_cos2theta_and_pdf_array(cos2theta_pdf_function, num_pts=num_pts,
                                                               make_plot=make_plots, block=False)
    if debugging:
        print(print_preamble(), 'got power, cos2theta, and pdf arrays')
    x0_denominator_values_to_try = linspace(start_test_value, end_test_value, num_test_values)
    desired_cdf_value = CL_percentage / 100
    x0_numerator = norm.ppf(desired_cdf_value)
    print(print_preamble(), 'x0 numerator for CL of', CL_percentage, 'is', x0_numerator)

    # now make up the pdf 2D arrays
    joint_pdf_values = zeros((len(x0_denominator_values_to_try), num_pts, num_pts))
    for k, (c, p) in enumerate(zip(c_array, cos2theta_pdf_array)):  # using array-valued inputs to make this run faster
        # normalized_power_pdf = norm.pdf([x_array], (x0_numerator / transpose([x0_denominator_values_to_try])) * c, 1)
        # joint_pdf_values[:, :, k] = p * normalized_power_pdf
        # the joint pdf is found by multiplying two pdfs together
        joint_pdf_values[:, :, k] = get_joint_pdf([x_array], c, p,
                                                  x0=(x0_numerator / transpose([x0_denominator_values_to_try])))
    print(print_preamble(), 'joint pdf arrays made')
    # get_joint_pdf(x, cos2theta, cos2theta_pdf, x0=1.645 / 0.024)
    # normalized_power_pdf = norm.pdf(x, x0 * cos2theta, 1)  # this is the PDF of a normalized gaussian random variable
    # return cos2theta_pdf * normalized_power_pdf

    # now integrate the joint PDFs over the possible cos2theta values + get CDF arrays
    cdf_array = []
    if make_plots:
        figure()
    for joint_pdf, cos2theta in zip(joint_pdf_values, x0_denominator_values_to_try):
        integrated_pdf = [trapz(p_array, c_array) for p_array in joint_pdf]
        if make_plots:
            plot(x_array, integrated_pdf, label=cos2theta)
        # integrate to 0 to get the CDF for a zero power excess
        cdf_array.append(trapz(integrated_pdf, x_array))

    if make_plots:
        xlabel('normalized power fluctuation')
        ylabel('PDF for $\\cos^2(\\theta)$')
        legend()
        show(block=False)

        figure()
        plot(x0_denominator_values_to_try, cdf_array)
        xlabel('$<\\cos^2(\\theta)>$')
        ylabel('CL of hidden photon exclusion')
        show(block=block)

    # finally use interpolation to calculate what cos2theta should be
    x0_function = interp1d(cdf_array, x0_denominator_values_to_try)  # for any CL, what x0 do we need?
    cos2theta_value = x0_function(1 - desired_cdf_value)
    print(print_preamble(), 'Using interpolation, the calculated value of cos2theta is', cos2theta_value,
          'for a CL of', CL_percentage)
    return cos2theta_value


def save_joint_pdf_arrays(joint_pdf_npz_file_name, x0_numerator_values_to_try, x0_denominator_values_to_try,
                          cos2theta_pdf_function, num_pts=500):
    """
    Saves and returns the joint PDF arrays along with the values of normalized power and cos2theta used to calculate
    them and the labels associated with each joint PDF array for plotting purposes.
    :param joint_pdf_npz_file_name: name of file containing the joint PDF arrays calculated with cos2theta_pdf_function
    :param x0_numerator_values_to_try: calculated using norm.ppf(desired_cdf_value)
    :param x0_denominator_values_to_try: from find_cos2theta_from_joint_pdf() - these are candidate cos2theta values
    :param cos2theta_pdf_function: which function to use for the PDF of cos2theta, dependent on timing information.
    :param num_pts: Number of points for the simulation. You can input a lower number if your computer is slow.
    :return: the joint PDFs, the arrays for normalized power and cos2theta, and labels for each joint PDF
    """
    # first the two random variables
    x_array = linspace(-5, 80, num_pts)  # possible values of the normalized power fluctuation
    c_array, cos2theta_pdf_array = get_cos2theta_and_pdf_array(cos2theta_pdf_function)

    # now potential desired values of cos2theta and our CL, calculated using find_cos2theta_from_joint_pdf()
    labels_for_each_curve = ['standard gaussian for 90% CL', 'standard gaussian for 95% CL']
    labels_for_each_curve.extend(['x0 = ' + str(num) + ' / ' + str(den)
                                  for num, den in zip(x0_numerator_values_to_try, x0_denominator_values_to_try)])
    # now make up the pdf 2D arrays
    joint_pdf_values = zeros((len(x0_denominator_values_to_try) + 2, num_pts, num_pts))
    for i, x in enumerate(x_array):
        for j, (c, p) in enumerate(zip(c_array, cos2theta_pdf_array)):
            joint_pdf_values[0, i, j] = norm.pdf(x, 1.282, 1)  # for the 90% CL
            joint_pdf_values[1, i, j] = norm.pdf(x, 1.645, 1)  # for the 95% CL
            for k, (numerator, denominator) in enumerate(zip(x0_numerator_values_to_try, x0_denominator_values_to_try)):
                joint_pdf_values[k + 2, i, j] = get_joint_pdf(x, c, p, x0=numerator / denominator)

    # finally save the arrays so that they can be called in make_joint_pdf_plots()
    savez(joint_pdf_npz_file_name + '.npz', joint_pdf_arrays=joint_pdf_values,
          cos2theta_array=c_array, normalized_power_array=x_array, labels_for_each_curve=labels_for_each_curve)
    return joint_pdf_values, c_array, x_array, labels_for_each_curve


def save_instantaneous_joint_pdf_arrays(file_name='joint_pdf_arrays', re_calculate_test_value=False,
                                        make_plots=False, block=True):
    """
    Saves the numerically calculated joint PDFs between {the normalized power fluctuation out of the haloscope} and {the
    dot product between the polarization of the hidden photon and the axis of the haloscope} for an instantaneous hidden
    photon measurement
    :param file_name: string for the name of the file to save the joint pdf arrays to
    :param re_calculate_test_value: boolean on whether to re-calculate cos2theta or to use the hardcoded values
    :param make_plots: boolean on whether to make a plot of the cos2theta PDF
    :param block: if plots are made, boolean option to require all plots be closed before the next bit of code runs
    :return: name of the file that the joint pdf arrays are saved to
    """
    cos2theta_pdf_function = get_instantaneous_cos2theta_pdf
    if re_calculate_test_value:
        test_value_90 = find_cos2theta_from_joint_pdf(cos2theta_pdf_function, 90,
                                                      start_test_value=0.01, end_test_value=0.1,
                                                      make_plots=make_plots, block=block)
        test_value_95 = find_cos2theta_from_joint_pdf(cos2theta_pdf_function, 95,
                                                      start_test_value=0.01, end_test_value=0.1,
                                                      make_plots=make_plots, block=block)
        cos2theta_values = [test_value_90, test_value_95, 0.0025, 0.0025]
    else:
        cos2theta_values = [0.076, 0.024, 0.0025, 0.0025]  # from find_cos2theta_from_joint_pdf()
    numerator_values = [1.282, 1.645, 1.282, 1.645]  # calculated using norm.ppf(desired_cdf_value)
    save_joint_pdf_arrays(file_name, numerator_values, cos2theta_values, cos2theta_pdf_function)
    print(print_preamble(), 'joint pdf arrays for instantaneous measurements have been saved')
    return file_name


def save_CAPP_15hr_joint_pdf_arrays(file_name='CAPP_spike_pdf_arrays', re_calculate_test_value=False,
                                    make_plots=False, block=True):
    """
    Saves the numerically calculated joint PDFs between {the normalized power fluctuation out of the haloscope} and {the
    dot product between the polarization of the hidden photon and the axis of the haloscope} for the 15-hour measurement
    that CAPP took at 2.59 GHz / 10.72 ueV to hit KSVZ for the axion measurement
    :param file_name: string for the name of the file to save the joint pdf arrays to
    :param re_calculate_test_value: boolean on whether to re-calculate cos2theta or to use the hardcoded values
    :param make_plots: boolean on whether to make a plot of the cos2theta PDF
    :param block: if plots are made, boolean option to require all plots be closed before the next bit of code runs
    :return: name of the file that the joint pdf arrays are saved to
    """
    try:
        npz_file = load('CAPP_pdf_to_interpolate.npz')
        pdf_x_axis_values = npz_file['pdf_x_axis_values']
        pdf_y_axis_values = npz_file['pdf_y_axis_values']
        cos2theta_pdf_function = interp1d(pdf_x_axis_values, pdf_y_axis_values)
    except FileNotFoundError:
        time_days = 15 / 24  # 15 hours
        latitude = 36.35  # Daejeon, Republic of Korea
        cos2theta_pdf_function = get_time_averaged_cos2th_pdf_function(time_days, latitude,
                                                                       make_plots=make_plots, block=False)
    print(print_preamble(), 'PDF for cos2theta found')
    if re_calculate_test_value:
        test_value = find_cos2theta_from_joint_pdf(cos2theta_pdf_function, 90, start_test_value=0.1, end_test_value=0.9,
                                                   make_plots=make_plots, block=block)
    else:
        test_value = 0.29
    print(print_preamble(), 'cos2theta found for CAPP to be', test_value)
    numerator_values = [1.282, 1.282, 1.282, 1.282]  # calculated using norm.ppf(desired_cdf_value)
    cos2theta_values = [test_value, 0.076, 0.024, 0.0025]  # from find_cos2theta_from_joint_pdf()
    save_joint_pdf_arrays(file_name, numerator_values, cos2theta_values, cos2theta_pdf_function)
    print(print_preamble(), 'joint pdf arrays for CAPP have been saved')
    return file_name


def make_joint_pdf_plots(joint_pdf_npz_file_name, number_of_joint_pdfs_to_plot=2, block=True):
    """
    Plots the joint PDFs between cos2theta (the square of the dot product of the hidden photon polarization) and the 
    axis of the cavity, the PDFs integrated over all values of cos2theta, and the corresponding CDFs as a function of 
    the normalized power fluctuations.
    :param joint_pdf_npz_file_name: .npz file containing the joint PDF arrays calculated with cos2theta_pdf_function
    :param number_of_joint_pdfs_to_plot: number of joint pdfs to plot, skipping the standard gaussians
    :param block: boolean option to require all plots be closed before the next bit of code runs
    :return: nothing
    """
    # first load the npz file with the joint pdf arrays
    npz_file = load(joint_pdf_npz_file_name + '.npz')
    joint_pdf_values = npz_file['joint_pdf_arrays']
    labels_for_each_curve = npz_file['labels_for_each_curve']
    x_array = npz_file['normalized_power_array']
    c_array = npz_file['cos2theta_array']

    # plot the joint PDF for the correct value of cos2theta for each CL, where cos2theta is the denominator of x0
    random_variable_parameter_limits = [c_array[0], c_array[-1], x_array[0], x_array[-1]]
    for i in range(number_of_joint_pdfs_to_plot):
        joint_pdf_to_plot = joint_pdf_values[i + 2]
        joint_pdf_to_plot[joint_pdf_to_plot <= 0.0001] = 0.0001  # so the logscale colorbar looks nice
        figure()
        imshow(joint_pdf_to_plot, aspect='auto', origin='lower', norm=LogNorm(),
               extent=random_variable_parameter_limits)
        colorbar()
        xlabel('$\\cos(\\theta)$')
        ylabel('Normalized power fluctuations')
        title('Joint PDF for ' + labels_for_each_curve[i + 2])
        show(block=False)

    # calculate the pdfs and cdfs
    integrated_pdfs = []
    integrated_cdfs = []
    for joint_pdf in joint_pdf_values:
        integrated_pdf = [trapz(p_array, x=c_array) for p_array in joint_pdf]
        integrated_cdf = cumtrapz(integrated_pdf, x_array)
        integrated_pdfs.append(integrated_pdf)
        integrated_cdfs.append(integrated_cdf)

    # plot pdfs
    figure()
    for pdf, label in zip(integrated_pdfs, labels_for_each_curve):
        plot(x_array, pdf, label=label)
    xlabel('normalized power')
    ylabel('joint pdf')
    legend()
    show(block=False)

    # finally, plot cdfs
    figure()
    for cdf, label in zip(integrated_cdfs, labels_for_each_curve):
        plot(x_array[1:], cdf, label=label)
    plot([x_array[0], x_array[-1]], [0.05, 0.05], 'k--')
    plot([x_array[0], x_array[-1]], [0.10, 0.10], 'k--')
    vlines(0, 0, 1, colors='k', linestyles='--')
    xlabel('normalized power')
    ylabel('joint cdf')
    legend()
    show(block=block)


def main(run_CAPP_or_instantaneous='CAPP'):  # function for what you're running
    if run_CAPP_or_instantaneous == 'CAPP':
        print(time() - start_time, 'Calculating cos2theta for CAPP')
        CAPP_file_name = save_CAPP_15hr_joint_pdf_arrays(make_plots=True, block=False)
        make_joint_pdf_plots(CAPP_file_name, number_of_joint_pdfs_to_plot=1)
    elif run_CAPP_or_instantaneous == 'instantaneous':
        print(time() - start_time, 'Calculating cos2theta for an instantaneous delta function measurement')
        file_name = save_instantaneous_joint_pdf_arrays(re_calculate_test_value=True, make_plots=True, block=False)
        make_joint_pdf_plots(file_name)
    else:
        print(time() - start_time, 'Please choose either CAPP or instantaneous to calculate the optimal <cos2theta>')


if __name__ == "__main__":  # if this is the script you run, and you're not just calling this script on another script
    main()
