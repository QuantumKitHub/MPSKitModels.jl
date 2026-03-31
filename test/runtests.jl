using ParallelTestRunner
using MPSKitModels

testsuite = ParallelTestRunner.find_tests(@__DIR__)

ParallelTestRunner.runtests(MPSKitModels, ARGS; testsuite)
