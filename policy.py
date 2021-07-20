# TODO: write policy function!!

###Test of the form_coalitions function!
# currentPhasemap = [1,2,3,4,5,3,2,3,5]
# elementCoalitions = [0,1,2,3,4,5,6,7,8]
#
# def form_coalitions():
#     # perhaps here we should round the phasemap values to a certain degree to form fewer coalitions?
#     if set(currentPhasemap) != set(elementCoalitions):
#         for elem in set(currentPhasemap):
#             coalition_for_elem_to_join = elementCoalitions[currentPhasemap.index(elem)]
#             for i in range(coalition_for_elem_to_join, len(elementCoalitions)):
#                 if currentPhasemap[i] == elem:
#                     elementCoalitions[i] = coalition_for_elem_to_join
#
# form_coalitions()
# print(elementCoalitions)