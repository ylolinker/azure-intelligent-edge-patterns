from __future__ import absolute_import, unicode_literals
from celery import shared_task
from .models import Project
from celery.utils.log import get_task_logger
import time

logger = get_task_logger(__name__)


# @shared_task(name="export_onnx_task")
@shared_task()
def export_onnx_task(pk):
    project_obj = Project.objects.get(pk=pk)
    trainer = project_obj.setting.revalidate_and_get_trainer_obj()
    customvision_project_id = project_obj.customvision_project_id
    camera = project_obj.camera
    while True:
        time.sleep(1)
        iterations = trainer.get_iterations(customvision_project_id)
        if len(iterations) == 0:
            logger.error('failed: not yet trained')
            return

        iteration = iterations[0]
        if iteration.exportable == False or iteration.status != 'Completed':
            continue

        exports = trainer.get_exports(
            customvision_project_id, iteration.id)
        if len(exports) == 0 or not exports[0].download_uri:
            logger.info('Status: exporting model')
            res = project_obj.export_iterationv3_2(iteration.id)
            logger.info(res.json())
            continue

        project_obj.download_uri = exports[0].download_uri
        project_obj.save()
        break
    return
