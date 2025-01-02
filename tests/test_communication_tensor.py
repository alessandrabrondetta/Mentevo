import numpy as np

from mentevo.utils import build_communication_tensor, build_forward_matrix
from .utils import epsilon_equal


def _original_A_matrix(Na, No, alpha, beta, gamma, delta, Ao, Aa):
    return np.kron(np.eye(Na), alpha * np.eye(No) + beta * Ao) + np.kron(Aa, gamma * np.eye(No) + delta * Ao)


def _original_F(Na, No, A, z):
    inds = np.kron(np.ones((Na, Na)), np.eye(No)) > 0
    A_SO = np.zeros((Na * No, Na * No))
    A_SO[inds] = A[inds]
    A_SO = A_SO
    # assemble same-option interactions and bias (2)
    F1 = (np.dot(A_SO, z))

    F2 = np.zeros_like(F1)
    # assemble inter-option interactions about option j (3)
    for j in range(No):
        A_j = np.zeros((Na * No, Na * No))
        ind_mat = np.zeros((No, No))
        ind_mat[:, j] = np.ones(No)
        ind_mat[j, j] = 0
        ind_mat2 = np.kron(np.ones((Na, Na)), ind_mat)
        A_j[ind_mat2 > 0] = A[ind_mat2 > 0]
        F2 += np.dot(A_j, z)

    F = F1 + F2

    return F, F1, F2


def test_forward_matrix_1_agent():
    na = 1
    no = 2
    gamma = 1.0
    delta = 1.0

    task_graph = np.array([[1, -1], [-1, 1]])
    communication_graph = np.array([[1]])

    for alpha in [1.0, 10.0, 5.0]:
        for beta in [0.1, 0.5, 0.9]:

            F = build_forward_matrix(na, no, alpha, beta, gamma, delta, task_graph, communication_graph)

            assert F.shape == (na * no, na * no)
            assert epsilon_equal(F, np.array([
                [alpha, -beta],
                [-beta, alpha],
            ]))


def test_forward_matrix_2_agent():
    na = 2
    no = 2
    gamma = 1.0
    delta = 1.0

    task_graph = np.array([[1, -1], [-1, 1]])
    communication_graph = np.array([[1, 1], [1, 1]])

    for alpha in [1.0, 10.0, 5.0]:
        for beta in [0.1, 0.5, 0.9]:
            for gamma in [0.1, 0.3, 1.0]:
                for delta in [0.1, 0.3, 1.0]:

                    F = build_forward_matrix(na, no, alpha, beta, gamma, delta, task_graph, communication_graph)

                    assert F.shape == (na * no, na * no)
                    assert epsilon_equal(F, np.array([
                        [alpha, -beta, gamma, -delta],
                        [-beta, alpha, -delta, gamma],
                        [gamma, -delta, alpha, -beta],
                        [-delta, gamma, -beta, alpha],
                    ]))


def test_forward_communication_graph():
    na = 2
    no = 2
    alpha = 1.0
    beta = 1.0
    gamma = 1.0
    delta = 1.0

    task_graph = np.array([[1, -1], [-1, 1]])
    # everything is connected to everything
    c0 = np.array([[1, 1], [1, 1]])
    F = build_forward_matrix(na, no, alpha, beta, gamma, delta, task_graph, c0)

    assert F.shape == (na * no, na * no)
    assert epsilon_equal(F, np.array([
        [alpha, -beta, gamma, -delta],
        [-beta, alpha, -delta, gamma],
        [gamma, -delta, alpha, -beta],
        [-delta, gamma, -beta, alpha],
    ]))

    # now cut the connection between the agents
    c1 = np.array([[1, 0], [0, 1]])
    F = build_forward_matrix(na, no, alpha, beta, gamma, delta, task_graph, c1)

    assert F.shape == (na * no, na * no)
    assert epsilon_equal(F, np.array([
        [alpha, -beta, 0, 0],
        [-beta, alpha, 0, 0],
        [0, 0, alpha, -beta],
        [0, 0, -beta, alpha],
    ]))

    # now cut the intra-agent connection
    c2 = np.array([[0, 1], [1, 0]])
    F = build_forward_matrix(na, no, alpha, beta, gamma, delta, task_graph, c2)

    assert F.shape == (na * no, na * no)
    assert epsilon_equal(F, np.array([
        [0, 0, gamma, -delta],
        [0, 0, -delta, gamma],
        [gamma, -delta, 0, 0],
        [-delta, gamma, 0, 0],
    ]))


def test_task_graph():
    na = 2
    no = 2
    alpha = 1.0
    beta = 1.0
    gamma = 1.0
    delta = 1.0

    communication_graph = np.array([[1, 1], [1, 1]])
    # classical task graph
    t0 = np.array([[1, -1], [-1, 1]])

    F = build_forward_matrix(na, no, alpha, beta, gamma, delta, t0, communication_graph)

    assert F.shape == (na * no, na * no)
    assert epsilon_equal(F, np.array([
        [alpha, -beta, gamma, -delta],
        [-beta, alpha, -delta, gamma],
        [gamma, -delta, alpha, -beta],
        [-delta, gamma, -beta, alpha],
    ]))

    # now cut the connection between the tasks
    t1 = np.array([[1, 0], [0, 1]])
    F = build_forward_matrix(na, no, alpha, beta, gamma, delta, t1, communication_graph)

    assert F.shape == (na * no, na * no)
    assert epsilon_equal(F, np.array([
        [alpha, 0, gamma, 0],
        [0, alpha, 0, gamma],
        [gamma, 0, alpha, 0],
        [0, gamma, 0, alpha],
    ]))

    # now cut the intra-task connection
    t2 = np.array([[0, -1], [-1, 0]])
    F = build_forward_matrix(na, no, alpha, beta, gamma, delta, t2, communication_graph)

    assert F.shape == (na * no, na * no)
    assert epsilon_equal(F, np.array([
        [0, -beta, 0, -delta],
        [-beta, 0, -delta, 0],
        [0, -delta, 0, -beta],
        [-delta, 0, -beta, 0],
    ]))


def test_forward_matrix_3_agent():
    na = 3
    no = 2

    task_graph = np.array([[1, -1], [-1, 1]])
    communication_graph = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    for alpha in [1.0, 10.0, 5.0]:
        for beta in [0.1, 0.5, 0.9]:
            for gamma in [0.1, 0.3, 1.0]:
                for delta in [0.1, 0.3, 1.0]:

                    F = build_forward_matrix(na, no, alpha, beta, gamma, delta, task_graph, communication_graph)

                    assert F.shape == (na * no, na * no)
                    assert epsilon_equal(F, np.array([
                        [alpha, -beta, gamma, -delta, gamma, -delta],
                        [-beta, alpha, -delta, gamma, -delta, gamma],
                        [gamma, -delta, alpha, -beta, gamma, -delta],
                        [-delta, gamma, -beta, alpha, -delta, gamma],
                        [gamma, -delta, gamma, -delta, alpha, -beta],
                        [-delta, gamma, -delta, gamma, -beta, alpha],
                    ]))
