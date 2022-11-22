/* Shared Use License: This file is owned by Derivative Inc. (Derivative)
* and can only be used, and/or modified for use, in conjunction with
* Derivative's TouchDesigner software, and only if you are a licensee who has
* accepted Derivative's TouchDesigner license or assignment agreement
* (which also govern the use of this file). You may share or redistribute
* a modified version of this file provided the following conditions are met:
*
* 1. The shared file or redistribution must retain the information set out
* above and this list of conditions.
* 2. Derivative's name (Derivative Inc.) or its trademarks may not be used
* to endorse or promote products derived from this file without specific
* prior written permission from Derivative.
*/

#include "sortParticles.h"

#include <assert.h>
#include <cstdio>
#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>
#include <format>

// These functions are basic C function, which the DLL loader can find
// much easier than finding a C++ Class.
// The DLLEXPORT prefix is needed so the compile exports these functions from the .dll
// you are creating
extern "C"
{
DLLEXPORT
void
FillTOPPluginInfo(TOP_PluginInfo *info)
{
	// This must always be set to this constant
	info->apiVersion = TOPCPlusPlusAPIVersion;

	// Change this to change the executeMode behavior of this plugin.
	info->executeMode = TOP_ExecuteMode::CUDA;

	// The opType is the unique name for this TOP. It must start with a 
	// capital A-Z character, and all the following characters must lower case
	// or numbers (a-z, 0-9)
	info->customOPInfo.opType->setString("Sortparticles");

	// The opLabel is the text that will show up in the OP Create Dialog
	info->customOPInfo.opLabel->setString("Sort Particles");

	// Will be turned into a 3 letter icon on the nodes
	info->customOPInfo.opIcon->setString("SPA");

	// Information about the author of this OP
	info->customOPInfo.authorName->setString("Josef Pelz");
	info->customOPInfo.authorEmail->setString("contact@josefpelz.com");

	// This TOP works with 0 or 1 inputs connected
	info->customOPInfo.minInputs = 1;
	info->customOPInfo.maxInputs = 1;
}

DLLEXPORT
TOP_CPlusPlusBase*
CreateTOPInstance(const OP_NodeInfo* info, TOP_Context *context)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per TOP that is using the .dll

	// Note we can't do any OpenGL work during instantiation

	return new sortParticles(info, context);
}

DLLEXPORT
void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context *context)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the TOP using that instance is deleted, or
	// if the TOP loads a different DLL

	// We do some OpenGL teardown on destruction, so ask the TOP_Context
	// to set up our OpenGL context

	delete (sortParticles*)instance;
}

};


sortParticles::sortParticles(const OP_NodeInfo* info, TOP_Context *context) :
	myNodeInfo(info), myExecuteCount(0),
	myError(nullptr),
	myInputSurface(0),
	myContext(context)
{
	myOutputSurfaces.fill(0);
}

sortParticles::~sortParticles()
{
	if (myInputSurface)
		cudaDestroySurfaceObject(myInputSurface);
	for (auto o : myOutputSurfaces)
	{ 
		if (o)
			cudaDestroySurfaceObject(o);
	}
}

void
sortParticles::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs *inputs, void* reserved)
{
	// Setting cookEveryFrameIfAsked to true causes the TOP to cook every frame
	// but only if something asks for it's output.
	ginfo->cookEveryFrameIfAsked = false;
}

extern cudaError_t sort(int width, int height, double3 reference, cudaSurfaceObject_t input, cudaSurfaceObject_t output);

static void
setupCudaSurface(cudaSurfaceObject_t* surface, cudaArray_t array)
{
	if (*surface)
	{
		cudaResourceDesc desc;
		cudaGetSurfaceObjectResourceDesc(&desc, *surface);
		if (desc.resType != cudaResourceTypeArray ||
			desc.res.array.array != array)
		{
			cudaDestroySurfaceObject(*surface);
			*surface = 0;
		}
	}

	if (!*surface)
	{
		cudaResourceDesc desc;
		desc.resType = cudaResourceTypeArray;
		desc.res.array.array = array;
		cudaCreateSurfaceObject(surface, &desc);
	}
}

void
sortParticles::execute(TOP_Output* output, const OP_Inputs* inputs, void* reserved)
{
	myError = nullptr;
	myExecuteCount++;
	TOP_CUDAOutputInfo info;
	info.textureDesc.pixelFormat = OP_PixelFormat::RGBA32Float;
	info.textureDesc.texDim = OP_TexDim::e2D;

	const OP_CUDAArrayInfo* inputArray = nullptr;
	if (inputs->getNumInputs() > 0)
	{
		const OP_TOPInput* topInput = inputs->getInputTOP(0);
		info.textureDesc.width = topInput->textureDesc.width;
		info.textureDesc.height = topInput->textureDesc.height;

		if (topInput->textureDesc.pixelFormat != OP_PixelFormat::RGBA32Float)
		{
			myError = "Input should have RGBA32 pixel format.";
			return;
		}

		if (topInput->textureDesc.texDim != OP_TexDim::e2D) {
			myError = "Input should be 2D texture";
			return;
		}

		OP_CUDAAcquireInfo acquireInfo;

		acquireInfo.stream = cudaStream_t(0);
		
		inputArray = topInput->getCUDAArray(acquireInfo, nullptr);
		if (!inputArray) {
			myError = "couldn't load input.";
			return;
		}
	}
	else {
		myError = "Requires one input.";
		return;
	}

	const OP_CUDAArrayInfo* outputInfo = output->createCUDAArray(info, nullptr);
	if (!outputInfo)
		return;

	// Output to a second color buffer, with a different resolution. Use a Render Select TOP
	// to get this output.
	// All calls to the 'inputs' need to be made before beginCUDAOperations() is called
	const OP_ObjectInput* referenceComp = inputs->getParObject("Reference");
	if (!referenceComp) {
		myError = "provide valid reference COMP.";
		return;
	}

	double3 reference = make_double3(referenceComp->worldTransform[0][3], referenceComp->worldTransform[1][3], referenceComp->worldTransform[2][3]);


	// Now that we have gotten all of the pointers to the OP_CUDAArrayInfos that we may want, we can tell the context
	// that we are going to start doing CUDA operations. This will cause the cudaArray members of the OP_CUDAArrayInfo
	// to get filled in with valid addresses.
	if (!myContext->beginCUDAOperations(nullptr))
		return;

	setupCudaSurface(&myOutputSurfaces[0], outputInfo->cudaArray);
	setupCudaSurface(&myInputSurface, inputArray->cudaArray);

	sort(info.textureDesc.width, info.textureDesc.height, reference, myInputSurface, myOutputSurfaces[0]);

	myContext->endCUDAOperations(nullptr);
}

int32_t
sortParticles::getNumInfoCHOPChans(void* reserved)
{
	// We return the number of channel we want to output to any Info CHOP
	// connected to the TOP. In this example we are just going to send one channel.
	return 1;
}

void
sortParticles::getInfoCHOPChan(int32_t index,
						OP_InfoCHOPChan* chan,
						void* reserved)
{
	// This function will be called once for each channel we said we'd want to return
	// In this example it'll only be called once.

	if (index == 0)
	{
		chan->name->setString("executeCount");
		chan->value = (float)myExecuteCount;
	}
}

bool		
sortParticles::getInfoDATSize(OP_InfoDATSize* infoSize, void* reserved)
{
	infoSize->rows = 1;
	infoSize->cols = 2;
	// Setting this to false means we'll be assigning values to the table
	// one row at a time. True means we'll do it one column at a time.
	infoSize->byColumn = false;
	return true;
}

void
sortParticles::getInfoDATEntries(int32_t index,
										int32_t nEntries,
										OP_InfoDATEntries* entries,
										void* reserved)
{
	char tempBuffer[4096];

	if (index == 0)
	{
		// Set the value for the first column
#ifdef _WIN32
		strcpy_s(tempBuffer, "executeCount");
#else // macOS
		strlcpy(tempBuffer, "executeCount", sizeof(tempBuffer));
#endif
		entries->values[0]->setString(tempBuffer);

		// Set the value for the second column
#ifdef _WIN32
		sprintf_s(tempBuffer, "%d", myExecuteCount);
#else // macOS
		snprintf(tempBuffer, sizeof(tempBuffer), "%d", myExecuteCount);
#endif
		entries->values[1]->setString(tempBuffer);
	}
}

void
sortParticles::getErrorString(OP_String *error, void* reserved)
{
	error->setString(myError);
}

void
sortParticles::setupParameters(OP_ParameterManager* manager, void* reserved)
{
	{
		OP_StringParameter	np;

		np.name = "Reference";
		np.label = "Reference";

		np.defaultValue = "";

		OP_ParAppendResult res = manager->appendCOMP(np);
		assert(res == OP_ParAppendResult::Success);
	}
}

void
sortParticles::pulsePressed(const char* name, void* reserved)
{
	if (!strcmp(name, "Reset"))
	{
		// Do something to reset here
	}
}
