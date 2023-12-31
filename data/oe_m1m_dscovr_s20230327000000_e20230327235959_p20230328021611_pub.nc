CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230327000000_e20230327235959_p20230328021611_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-28T02:16:11.666Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-27T00:00:00.000Z   time_coverage_end         2023-03-27T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxr�@  T          @�{@�ff�E�@   A��HC�Ф@�ff�{�>��@x��C��                                    Bxr��  �          @�(�@��R�k�@#�
A���C�\)@��R��\)>L��?У�C��                                    Bxr�  �          @�z�@�
=�p��@(�A�Q�C�  @�
=��\)=u?   C���                                    Bxr2  
�          @��H@�
=�L��@#�
A��C���@�
=��=q>�ff@s33C���                                    Bxr�  
�          @ۅ@��\�A�@%A��C��H@��\�|(�?z�@�=qC�aH                                    Bxr~  
�          @�(�@��S�
@(Q�A�p�C�W
@���ff>�@u�C�(�                                    Bxr-$  �          @ۅ@���3�
@4z�A�ffC���@���xQ�?aG�@��C���                                    Bxr;�  
�          @ڏ\@�{��\@333A\C���@�{�?\)?�
=AB{C���                                    BxrJp  T          @ڏ\@�33��\)@?\)A�C��@�33�K�?��
AO�C��q                                    BxrY  �          @��H@��
�G�@5A��HC��f@��
�N{?�=qA4(�C��                                     Bxrg�  
          @ڏ\@�ff����@p�A���C�p�@�ff�!�?��A1�C��                                    Bxrvb  
g          @�=q@��Ϳ�{@'�A��\C��@����"�\?��RAJffC���                                    Bxr�  
�          @�33@�z���H@:=qA��HC�˅@�z��Mp�?�A?\)C��
                                    Bxr��  �          @ۅ@�p��@C�
A�=qC�Ǯ@�p��g
=?���A5p�C��                                     Bxr�T  	�          @ۅ@����%@G
=A��
C�p�@����vff?�G�A(��C��{                                    Bxr��  a          @�p�@����C�
@O\)A߅C��@������?�{A��C�^�                                    Bxr��  �          @�G�@�(��tz�@S33A�z�C���@�(����?E�@��C�1�                                    Bxr�F  
(          @��@�Q���  @VffA�=qC���@�Q����?5@�G�C�XR                                    Bxr��  �          @ᙚ@�������@Mp�A���C�@������?z�@�Q�C���                                    Bxr�  
�          @��@�{��G�@C�
A��
C��f@�{��Q�>��
@&ffC���                                    Bxr�8  
�          @�\@�(���Q�@L��A�G�C�w
@�(���\)>B�\?\C��H                                    Bxr�  "          @�=q@x����Q�@FffA�(�C��@x����z���uC���                                    Bxr�  
�          @��@n{���@C33AɅC���@n{���
�k�����C���                                    Bxr&*  
M          @���@�  ��
=@<��A£�C��{@�  ��=q    <#�
C�}q                                    Bxr4�  
�          @�z�@�\)���
@C33A�Q�C�~�@�\)���<#�
=uC�/\                                    BxrCv  T          @�(�@Vff���R@Mp�A��C��=@Vff�ʏ\�#�
���C���                                    BxrR  T          @�33@P����p�@/\)A�  C��@P���ȣ׿+���ffC���                                    Bxr`�  
�          @��
@Z�H���@O\)A��C�g�@Z�H��
=�u���C�j=                                    Bxroh  
�          @��
@L����G�@(��A�=qC�z�@L����녿Q��ҏ\C�t{                                    Bxr~  �          @�(�@`  ���
@$z�A�p�C���@`  ��(��J=q���C���                                    Bxr��  
�          @�(�@N{���
@
=A�
=C�\)@N{��\)������RC���                                    Bxr�Z  
�          @ᙚ?�
=��
=?�Q�A�{C�L�?�
=�У׿�\�ip�C�>�                                    Bxr�   
�          @�=q@B�\��  ?�(�A�ffC�k�@B�\��(���  �EG�C�*=                                    Bxr��  �          @�33@Z=q��ff@\)A�(�C�l�@Z=q����fff��  C�y�                                    Bxr�L  �          @�@J=q��p�@33A�G�C�@J=q��  ����C�n                                    Bxr��  T          @�=q@�������@7
=A�C��@�����녽�Q�5C��)                                    Bxr�  �          @�G�@��
���@<(�A��C��@��
��G�>���@-p�C��                                    Bxr�>  "          @��@�ff�|��@,(�A�
=C�` @�ff����>W
=?�z�C��
                                    Bxr�  �          @���@�Q����
@.{A�C�G�@�Q���{>\)?�33C��R                                    Bxr�  �          @��@����vff@(��A�p�C���@������>aG�?�\C�T{                                    Bxr0  �          @�=q@�G����\@1G�A���C��@�G���p�>W
=?�(�C��
                                    Bxr-�  �          @��H@�\)��33@�A��RC���@�\)��Q콸Q�5C��3                                    Bxr<|  T          @��H@����p�@?\)A��C��=@������=�Q�?B�\C�:�                                    BxrK"  �          @�z�@��\��=q@&ffA��HC�P�@��\��  �8Q쿼(�C�P�                                    BxrY�  "          @��@����=q@2�\A��C�@��������
�8Q�C�                                    Bxrhn  �          @�(�@������\@@  A�p�C�h�@�����
=>��?��RC��                                     Bxrw  
�          @��
@���ff@>�RA��C�9�@���33>L��?У�C��R                                    Bxr��  a          @��
@�(����@;�A��C���@�(���p�=���?O\)C�=q                                    Bxr�`  �          @ᙚ@���~�R@HQ�A��
C�W
@������?z�@�Q�C��                                    Bxr�  T          @޸R@��\��p�@H��A���C���@��\��{>��H@�=qC��{                                    Bxr��  �          @���@������@A�AЏ\C�4{@����
=>�33@9��C�S3                                    Bxr�R  "          @�@��
��Q�@<(�Aȣ�C��@��
��(�>��?�(�C��=                                    Bxr��  �          @��@�{��33@?\)A͙�C��\@�{����>�\)@z�C��                                    Bxrݞ  �          @�z�@�(��~{@K�A܏\C��@�(�����?#�
@��C�@                                     Bxr�D  "          @�p�@����z�@H��A�Q�C�:�@����(�>Ǯ@Mp�C�U�                                    Bxr��  �          @�
=@xQ���z�@EA���C�9�@xQ�����=�Q�?@  C��                                    Bxr		�  �          @�(�@�����
@AG�A��HC���@������>#�
?�\)C�/\                                    Bxr	6  "          @߮@s33��33@=qA�\)C���@s33��=q�:�H���C���                                    Bxr	&�  �          @޸R@�ff���H@)��A�C�|)@�ff��Q쾅���C��=                                    Bxr	5�  �          @ۅ@�p���z�@L(�A��C�n@�p���ff?�@�\)C�1�                                    Bxr	D(  �          @�z�@�z���  @L(�Aܣ�C��3@�z���G�?�\@���C���                                    Bxr	R�  �          @�=q@��\�p  @<(�A�G�C�R@��\��\)?\)@��C��                                    Bxr	at  
�          @���@�z��W�@)��A��RC�H@�z���  ?�\@�  C��                                     Bxr	p            @�(�@�p��Y��@p�A���C��@�p���{>�{@6ffC�&f                                    Bxr	~�  /          @ۅ@���g
=@<(�A�C��@�����?#�
@��HC��f                                    Bxr	�f  �          @��H@���,��@ffA��C���@���a�?(�@��C�y�                                    Bxr	�  T          @��@���#33@'
=A��C�@ @���a�?h��@��HC�w
                                    Bxr	��  y          @��H@�����@#33A�G�C�:�@����S�
?z�HA\)C�W
                                    Bxr	�X  �          @�G�@ȣ׿��@33A���C�)@ȣ��Q�?uA{C���                                    Bxr	��  �          @��H@�  ��Q�@2�\A��RC�#�@�  �8��?�G�AL��C�&f                                    Bxr	֤  �          @�(�@�{�{@(��A�ffC���@�{�QG�?�{A(�C���                                    Bxr	�J  T          @��H@�\)�p�@{A�G�C�޸@�\)�J=q?xQ�A�HC�                                    Bxr	��  �          @��
@���G
=@5�A���C��@����(�?Q�@ڏ\C�W
                                    Bxr
�  �          @�p�@���@*=qA�\)C�z�@��Tz�?���A�RC�]q                                    Bxr
<  T          @�z�@���*�H@<(�Aʏ\C�B�@���s33?��A��C��                                    Bxr
�  �          @ۅ@����S33@:=qA�z�C���@�����=q?J=q@��
C�P�                                    Bxr
.�  T          @�ff@�Q��Q�@C33AиRC�@�Q��g�?��A8��C���                                    Bxr
=.  �          @߮@����x��@Dz�A�33C���@�����p�?!G�@�{C�q�                                    Bxr
K�  "          @�Q�@�{�o\)@P��A�(�C�o\@�{��(�?^�R@�p�C���                                    Bxr
Zz  
�          @߮@�33�y��@H��A�=qC���@�33���R?0��@�p�C�+�                                    Bxr
i   T          @�Q�@�
=���@Z�HA��C���@�
=��=q?L��@���C��                                    Bxr
w�  �          @���@�p��qG�@j=qA��HC���@�p����?��HA{C�9�                                    Bxr
�l  z          @�Q�@���hQ�@P��Aޣ�C���@������?p��@��RC��                                    Bxr
�  �          @޸R@����G�@�
A�G�C��3@����w
=>Ǯ@L��C��                                    Bxr
��  �          @߮@�ff�mp�@N�RA�Q�C���@�ff���H?^�R@��C��3                                    Bxr
�^  �          @�Q�@�33�z�H@K�A�  C��H@�33��  ?8Q�@�p�C�\                                    Bxr
�  �          @���@�����H@L��AظRC��q@������?(��@�z�C�P�                                    Bxr
Ϫ  T          @���@�33��  @L��A�ffC��q@�33��G�?�@�{C��                                     Bxr
�P  z          @���@����H@S33A��C��\@����?�R@���C���                                    Bxr
��  �          @���@�ff��\)@C�
A��HC�c�@�ff��p�>�{@3�
C���                                    Bxr
��  �          @޸R@�33���@B�\Aϙ�C��@�33���>���@-p�C�h�                                    Bxr
B  �          @�Q�@��R��33@L��A�G�C���@��R���
?�@��
C��R                                    Bxr�  �          @߮@�ff��
=@VffA���C��)@�ff����?��@�{C��\                                    Bxr'�  T          @�  @�G�����@G�A�  C��@�G����\?!G�@��C���                                    Bxr64  
�          @�  @�(����\@Q�A�(�C�O\@�(���?@  @�{C���                                    BxrD�  �          @�
=@�33�w�@G�A�C��
@�33��p�?8Q�@�\)C�L�                                    BxrS�  T          @�
=@��s�
@FffA�C�&f@����?=p�@��HC���                                    Bxrb&  z          @�\)@��R�QG�@4z�A���C���@��R��  ?E�@��C�                                    Bxrp�  �          @�Q�@�=q�J=q@5�A���C�7
@�=q���?W
=@ۅC��                                     Bxrr  �          @�  @���U@VffA��C��@�����\?�Q�A��C�33                                    Bxr�  �          @�Q�@����E@g
=A�{C��f@������?�ffALz�C�t{                                    Bxr��  �          @�G�@�  �~{@W�A�ffC��@�  ��(�?fff@�=qC�j=                                    Bxr�d  �          @ᙚ@����w
=@P��A�ffC��@�����\)?\(�@�  C�G�                                    Bxr�
  �          @��H@�p����@O\)A���C���@�p����>�G�@eC�1�                                    BxrȰ  
�          @�@K���(�@<��A�ffC��
@K���33��z��
=C�H�                                    Bxr�V  T          @�33@[����@N�RA؏\C�g�@[���
==�G�?c�
C�l�                                    Bxr��  �          @��@q���ff@P  A�=qC��\@q����>��R@!�C�L�                                    Bxr��  �          @��@�{���
@O\)A�33C�Y�@�{��{?8Q�@�z�C�                                    BxrH  T          @�\@_\)���@HQ�A�C��\@_\)���=#�
>�z�C���                                    Bxr�  �          @�33@n�R����@I��A��C��{@n�R��G�>\)?�\)C�ٚ                                    Bxr �  �          @�33@b�\����@I��A��C��\@b�\��z�=��
?�RC���                                    Bxr/:  �          @��H@j=q��{@HQ�A��
C���@j=q����=���?Tz�C��3                                    Bxr=�  �          @�z�@j�H���R@Mp�AծC��
@j�H�Å>#�
?��C�}q                                    BxrL�  �          @�\@c33��
=@c33A�33C���@c33�\?z�@��RC��                                    Bxr[,  T          @�33@s�
��{@@��A�Q�C�'�@s�
��  <#�
=uC�9�                                    Bxri�  �          @�33@q����@EA�Q�C��@q���Q�=���?E�C��                                    Bxrxx  T          @��
@s33���
@G�A�z�C�J=@s33��  >\)?�33C�/\                                    Bxr�  T          @�\@L����=q@XQ�A�z�C��H@L����G�>�=q@��C�u�                                    Bxr��  �          @�@/\)���H@Z=qA�\)C�@/\)�љ�>#�
?�ffC�\)                                    Bxr�j  �          @�@"�\���H@c33A��\C�9�@"�\�Ӆ>���@��C��H                                    Bxr�  T          @�z�@/\)���@XQ�A�{C��@/\)���H=�G�?h��C�J=                                    Bxr��  �          @�=q@\(����@X��A�C�)@\(���=q>�
=@\(�C��                                    Bxr�\  �          @�\)@�
=��G�@`��A�C���@�
=��  ?��A��C�4{                                    Bxr�  �          @�{@�z��5@g
=A�ffC���@�z���Q�?�(�Ag
=C�.                                    Bxr��  �          @�p�@���5�@j�HA��C���@������?��
An=qC�.                                    Bxr�N  T          @ڏ\@����0  @VffA�G�C�9�@�������?�ffAQ��C�<)                                    Bxr
�  �          @��H@�����@G�A�Q�C�^�@���hQ�?��
AO�C�t{                                    Bxr�  �          @�(�@h������@P  A�\C���@h������>�ff@n�RC��                                    Bxr(@  
�          @�=q@|�����@Z�HA��\C�R@|����
=?uAG�C��                                    Bxr6�  �          @��@��H��
=@G�A�\)C�)@��H�1�@   A�C�@                                     BxrE�  �          @�G�@�(���@   A�\)C�j=@�(��N{?�ffA
=C���                                    BxrT2  T          @�z�@��R��\@P  A�33C�+�@��R�XQ�?���Au�C���                                    Bxrb�  T          @�=q@�\)��z�@Z�HA�Q�C�j=@�\)�Vff@�\A�
=C�L�                                    Bxrq~  �          @ڏ\@��׿z�H@a�A��RC�(�@����%@%A�  C��                                    Bxr�$  �          @޸R@��R��
=@s33BG�C�&f@��R�9��@.{A�ffC��                                    Bxr��  �          @�ff@�
=�k�@s�
B\)C�j=@�
=�+�@7�A�G�C�u�                                    Bxr�p  �          @߮@�zῪ=q@z=qB��C�b�@�z��Dz�@0  A���C���                                    Bxr�  �          @ᙚ@��\��@���B�C��R@��\�N�R@5�A��
C���                                    Bxr��  �          @��@�����\@x��B=qC���@����y��@G�A���C�3                                    Bxr�b  �          @�\)@�p���@qG�BQ�C���@�p��E@%A�
=C��R                                    Bxr�  �          @���@�z��QG�@I��A���C�Ǯ@�z���z�?�33A�HC���                                   Bxr�  �          @�p�@�{��G�@;�A���C��H@�{��p�>�@j=qC��{                                   Bxr�T  �          @��
@�33��z�@2�\A���C���@�33��ff>�z�@�C��H                                    Bxr�  
�          @�\@�{����@FffAυC�g�@�{����?=p�@���C�@                                     Bxr�  �          @�{@���QG�@UA�z�C���@����\)?���A0  C�`                                     Bxr!F  
�          @�G�@�ff�dz�@Z=qA�G�C��@�ff����?��RA#\)C�                                    Bxr/�  �          @��@�(���
=@333A�p�C��H@�(���ff<�>�=qC��
                                    Bxr>�  T          @��@�G��~�R@J�HA��C�'�@�G���Q�?W
=@�p�C��R                                    BxrM8  �          @��@�  ��z�@EA��C�޸@�  ���\?�@�p�C��                                    Bxr[�  �          @�Q�@�Q����H@E�A�\)C���@�Q���=q?5@��
C��
                                    Bxrj�  �          @�=q@����_\)@H��A��HC�H�@�����=q?��A	��C���                                    Bxry*  "          @�G�@�G��hQ�@8Q�A���C��@�G���=q?E�@ȣ�C���                                    Bxr��  �          @��@�����z�@�HA��C��@�����\)�����
C�,�                                    Bxr�v  T          @�G�@�ff��  @��A��C��{@�ff��{���H�}p�C���                                    Bxr�  T          @��@�p���Q�@(�A��C�@ @�p���
=��p��B�\C���                                    Bxr��  �          @߮@�������@z�A�(�C���@������R��Q��=p�C�T{                                    Bxr�h  �          @߮@������?�
=A��C���@�����
�����  C��                                    Bxr�  �          @�\)@�G���
=?�p�AfffC�4{@�G���{�^�R��
=C��
                                    Bxrߴ  �          @��@����{?��HA��C�aH@�����\����XQ�C�*=                                    Bxr�Z  �          @���@�����  ?�\)Aw�C���@�����Q�^�R����C��                                    Bxr�   �          @߮@z�H���R?��A{�C�ٚ@z�H��ff�s33���HC�E                                    Bxr�  �          @�  @l(����\@�A�33C��R@l(����
�c�
��G�C�\                                    BxrL  �          @��@\����Q�@ ��A�G�C�y�@\�����׿}p����C��                                    Bxr(�  
�          @�  @��R���H@
=A�Q�C�Q�@��R��Q���r�\C�"�                                    Bxr7�  �          @��@����Q�@�A��C��3@������������RC���                                    BxrF>  T          @��H@��\��{@{A�C�e@��\���;���Q�C�"�                                    BxrT�  �          @��
@����33?�
=A{�
C��@�����Ϳ:�H����C��                                    Bxrc�  
�          @�@�Q����?޸RA`  C�1�@�Q�����}p����RC��{                                    Bxrr0  
�          @�(�@��\����?�Ao
=C�Ф@��\����Tz����C�!H                                    Bxr��  �          @���@�����@��A���C�K�@�������aG����C�Ф                                    Bxr�|  �          @�Q�@�
=��{@��A�  C�  @�
=��ff�.{��\)C���                                    Bxr�"  z          @޸R@��H��33?�Q�A��C�9�@��H��
=����w
=C�q                                    Bxr��  �          @߮@����z�@��A�
=C�Ǯ@����33���R�#�
C�n                                    Bxr�n  �          @�Q�@�ff����@#�
A�C��@�ff��{>�?��C�3                                    Bxr�  T          @��@�=q���
@'
=A���C���@�=q���\>u?�Q�C���                                    Bxrغ  �          @�\)@�=q����@,��A�Q�C�=q@�=q��G�>\@E�C��q                                    Bxr�`  
�          @�  @���s33@<��A�Q�C���@�����?Q�@׮C�l�                                    Bxr�  T          @�ff@����]p�@^�RA���C�` @�����{?�Q�A@(�C�q                                    Bxr�  �          @�
=@��Vff@Mp�AۮC���@���
=?�G�A&�\C���                                    BxrR  �          @޸R@����@��@J=qA�Q�C�g�@�����z�?�\)A6�\C�>�                                    Bxr!�  "          @޸R@�=q�QG�@Dz�AѮC�8R@�=q���\?�
=A��C�p�                                    Bxr0�  �          @�
=@��H�Z=q@8��A�  C��{@��H���
?u@�(�C�\)                                    Bxr?D  �          @�  @�
=�fff@>{AȸRC��
@�
=��=q?p��@��C�h�                                    BxrM�  T          @�G�@���aG�@8��A��C�^�@�����R?h��@�{C�!H                                    Bxr\�  �          @߮@�Q��aG�@<(�A�p�C��@�Q����?u@��C��                                    Bxrk6  �          @�\)@�p��j�H@8��AîC�N@�p����H?W
=@�{C�0�                                    Bxry�  T          @�
=@����c33@0  A��HC�
@�����p�?J=q@�  C��                                    Bxr��  �          @�\)@�33�[�@8Q�A��HC��f@�33��(�?s33@��HC�Y�                                    Bxr�(  �          @߮@�G��j�H@,(�A�
=C���@�G���  ?+�@�G�C�Ф                                    Bxr��  �          @�Q�@�G��j=q@0��A��C��f@�G�����?@  @���C���                                    Bxr�t  z          @߮@���hQ�@-p�A��RC��3@����
=?8Q�@�p�C��3                                    Bxr�  T          @�  @���{�@&ffA���C�\)@����>��H@���C��                                     Bxr��  �          @��@�Q����\@4z�A�G�C�Ff@�Q����>\@G�C�f                                    Bxr�f  �          @���@����z�@>�RA�Q�C���@�����?   @��C�9�                                    Bxr�  T          @��@�Q���  @<(�A�  C��@�Q����H?�@��
C�R                                    Bxr��  �          @���@������H@0��A�p�C���@�����(�>�ff@l��C�u�                                    BxrX  T          @��@�z���
=@I��A�G�C��R@�z���{?\(�@�  C��                                     Bxr�  
�          @�Q�@�33��\)@J�HA�\)C���@�33��
=?aG�@�C��{                                    Bxr)�  T          @�Q�@�33���
@N�RA�33C�3@�33��z�?z�HA�C��f                                    Bxr8J  T          @���@�����
@=p�A�\)C��)@������?@  @��HC��                                    BxrF�  �          @�=q@������@5�A��\C��
@������?z�@�p�C��                                    BxrU�  T          @��@��R��ff@0��A�ffC�ٚ@��R��  ?
=q@�(�C�^�                                    Bxrd<  �          @�33@������H@"�\A��C�
@������>.{?�{C�/\                                    Bxrr�  �          @�=q@��H����@��A�(�C��@��H���
        C�B�                                    Bxr��  �          @�\@�����=q@#33A��RC���@�����ff=��
?!G�C��q                                    Bxr�.  �          @�\@�G���Q�@(Q�A�Q�C��
@�G���>8Q�?��HC��{                                    Bxr��  �          @�{@����33@-p�A��HC���@������>W
=?�33C���                                    Bxr�z  �          @��H@�p����H@,(�A���C�� @�p����>��
@#�
C��=                                    Bxr�   {          @��H@�
=��G�@�RA��\C�Ff@�
=���þ8Q쿾�RC��                                    Bxr��  �          @�33@��R�W�@=p�A�C���@��R��33?�{A��C�{                                    Bxr�l  �          @��@����
=@:=qA���C�Q�@������?z�@��C��                                     Bxr�  �          @�{@�G�����@7�A���C�S3@�G����\?�@�ffC��
                                    Bxr��  �          @�ff@�p���z�@8��A�33C�"�@�p���
=?��@�G�C���                                    Bxr^  �          @�p�@�=q���@1�A��RC��@�=q��G�?z�@��
C��3                                    Bxr  �          @�ff@�  ���@z�A���C���@�  ���    <��
C�
=                                    Bxr"�  �          @�@��
��33@!�A�33C��
@��
����>��R@\)C���                                    Bxr1P  �          @��@���g�@QG�A�p�C���@����ff?�ffA'�C�\                                    Bxr?�  �          @��@��R�g�@S�
Aܣ�C��q@��R��
=?��A-G�C���                                    BxrN�  �          @�{@�G���(�@A�A�\)C�W
@�G�����?\(�@���C�z�                                    Bxr]B  �          @�ff@�{��Q�@?\)A�{C�"�@�{��p�?aG�@ᙚC�<)                                    Bxrk�  T          @�{@��H��  @2�\A�ffC�3@��H��G�?��@�G�C��                                     Bxrz�  �          @�ff@���w
=@AG�A�\)C�Ф@������?}p�@�C���                                    Bxr�4  �          @�ff@�\)���\@7
=A�\)C�� @�\)����?#�
@�=qC�                                    Bxr��  �          @�ff@������@C�
A��
C��f@������?J=q@�G�C��)                                    Bxr��  �          @�p�@��R��\)@>�RAď\C��f@��R���?L��@�(�C��                                    Bxr�&  T          @���@�G����\@@  A�z�C�� @�G����?aG�@�\C���                                    Bxr��            @���@�ff��z�@Dz�AˮC��@�ff��=q?n{@�{C�&f                                    Bxr�r  �          @��@����
=@I��AиRC��=@����p�?xQ�@�  C��q                                    Bxr�  T          @�{@��\��=q@HQ�A�ffC�
@��\��Q�?h��@�\)C�G�                                    Bxr�  �          @�p�@�{��(�@L(�A��C�w
@�{���H?n{@�ffC��                                    Bxr�d  �          @��@�����Q�@N{A�C�
@�����  ?��\A�\C�!H                                    Bxr
  �          @�@�\)����@P  AׅC��@�\)��G�?�z�A(�C�O\                                    Bxr�  �          @�p�@����o\)@B�\AȸRC�ff@�����ff?��A  C�=q                                    Bxr*V  T          @���@��R�q�@HQ�A�p�C�@��R��Q�?�z�A��C��                                    Bxr8�  T          @���@�  �q�@B�\A��C��@�  ��\)?�=qA
�\C��)                                    BxrG�  T          @�p�@����33@]p�A�C��R@����ff?�=qA+
=C�0�                                    BxrVH  T          @�
=@�ff��33@h��A��HC��
@�ff��  ?���A0��C�}q                                    Bxrd�  �          @�ff@�33����@W
=Aޏ\C�q@�33��?�{Ap�C�!H                                    Bxrs�  �          @�R@�ff��p�@Mp�AӅC�b�@�ff���
?xQ�@�
=C���                                    Bxr�:  T          @�
=@����Q�@:�HA��HC�o\@����=q?+�@��\C��                                    Bxr��  T          @�
=@�  ��@EA��C��@�  ���?xQ�@�  C�,�                                    Bxr��  �          @�
=@�����@E�A�=qC��@�����?��\A{C��{                                    Bxr�,  �          @�R@����qG�@E�Aʣ�C�K�@�����\)?�33A�\C�!H                                    Bxr��  T          @�
=@��
�w
=@8Q�A��\C�*=@��
��
=?p��@�RC�W
                                    Bxr�x  T          @�p�@��
�r�\@7�A���C�^�@��
���?s33@�(�C���                                    Bxr�  �          @���@����dz�@1G�A���C��\@������?u@�  C��                                    Bxr��  �          @�p�@���r�\@1�A��\C���@�����?aG�@��C�Ǯ                                    Bxr�j  
�          @�@����xQ�@+�A��C�(�@�������?@  @���C��H                                    Bxr  -          @�{@��
�vff@4z�A���C�+�@��
��{?fff@��C�o\                                    Bxr�  �          @�R@�(���p�@5A�Q�C�u�@�(���\)?J=q@���C��                                    Bxr#\  �          @�R@�=q���@@  AĸRC�K�@�=q����?n{@�{C��                                    Bxr2  �          @�{@������@FffA���C�O\@����ff?z�H@��HC��=                                    Bxr@�  T          @�ff@�{���R@EA��
C��@�{��(�?�  A Q�C���                                    BxrON  �          @�\)@��
���\@G�A�=qC�/\@��
��  ?z�H@�Q�C�q�                                    Bxr]�  T          @�ff@��R��p�@J=qA���C��)@��R��=q?aG�@�G�C�Z�                                    Bxrl�  �          @�R@�����p�@C33A�(�C���@�������?aG�@�G�C��                                    Bxr{@  �          @�\)@���p�@9��A��C���@����R?&ff@�C�P�                                    Bxr��  �          @�Q�@����{@7
=A�C��H@�����R?(�@�G�C�w
                                    Bxr��  "          @�Q�@�(���
=@>�RA�  C�� @�(����?O\)@�{C�H�                                    Bxr�2  �          @�\)@�����33@6ffA�C���@�������?@  @�ffC�!H                                    Bxr��  "          @�
=@�����@8��A�
=C�b�@�����
?z�@��C�O\                                    Bxr�~  �          @�@��
����@4z�A��
C��@��
��G�?
=q@��C��                                    Bxr�$  T          @�\)@�  ��z�@2�\A�C��@�  ���\>\@?\)C�
=                                    Bxr��  T          @�  @������H@3�
A���C�}q@�������>��
@   C���                                    Bxr�p  �          @�  @�Q���p�@+�A��C�=q@�Q�����>#�
?�G�C�Ǯ                                    Bxr�  T          @�@�33���H@(Q�A�z�C�� @�33��ff>��?�(�C�G�                                    Bxr�  T          @�@~�R��{@'�A�{C�)@~�R��G�=�G�?fffC���                                    Bxrb  �          @�@������@%A��
C�y�@����\)=���?Tz�C��                                    Bxr+  �          @�\)@�����H@'
=A�
=C���@����{>��?�C�XR                                    Bxr9�  �          @�R@��\��33@$z�A�33C��=@��\��=�G�?h��C�AH                                    BxrHT  �          @�@��
����@!G�A��RC���@��
���R=L��>��C�T{                                    BxrV�  T          @�
=@�����Q�@3�
A�(�C��3@������R>\@C33C�                                    Bxre�  T          @�\)@e��{@H��A͙�C��{@e��  ?(�@��C��                                    BxrtF  T          @�\)@qG����@2�\A�C�@ @qG�����>�=q@�C�                                    Bxr��  �          @�@hQ����H@333A�=qC�~�@hQ���  >�  ?�Q�C�3                                    Bxr��  �          @�
=@k�����@7
=A���C��
@k��ƸR>�{@,��C�Q�                                    Bxr�8  �          @�
=@�����33@,(�A���C�u�@������>�  ?��RC��3                                    Bxr��  T          @�R@u���p�@3�
A��
C�� @u���33>�{@.�RC��                                    Bxr��  �          @�Q�@|(���(�@4z�A�p�C��@|(����>�p�@:�HC��                                     Bxr�*  �          @�  @l(����H@0��A�\)C��
@l(���\)>u?�\)C�P�                                    Bxr��  �          @�  @e����\@:�HA�z�C�W
@e���G�>Ǯ@FffC��3                                    Bxr�v  �          @�G�@s33���@0  A���C�+�@s33��ff>u?�z�C��                                     Bxr�  T          @�Q�@�  ���\@�A���C���@�  ���������C��H                                    Bxr�  �          @�\)@��H��  @9��A�G�C�@��H����?8Q�@�{C���                                    Bxrh            @�p�@�33����@?\)A�
=C���@�33��p�?��A�C��                                    Bxr$  �          @�{@������@7�A�=qC���@�����H?aG�@ᙚC�8R                                    Bxr2�  �          @���@�p����@/\)A�=qC��f@�p����
?�R@�
=C��H                                    BxrAZ  �          @��@���p�@0��A�33C��@���z�?#�
@��C�|)                                    BxrP   �          @�(�@��\��Q�@*=qA��C��@��\��?�\@�z�C�\                                    Bxr^�  �          @�ff@�����@1G�A��HC��q@����(�?(��@�  C��\                                    BxrmL  �          @�\)@��\��\)@?\)A�33C��@��\��G�?Tz�@��
C���                                    Bxr{�  �          @�
=@�p���  @1G�A�ffC�=q@�p����R?!G�@��C�AH                                    Bxr��  "          @�ff@�(���G�@/\)A��RC�H@�(����?
=@�ffC�3                                    Bxr�>  T          @�  @�������@%A�G�C��3@������>�@i��C���                                    Bxr��  �          @�  @�(�����@)��A��C�,�@�(���=q?�@�Q�C�@                                     Bxr��  
�          @�R@�{���@&ffA���C�&f@�{��ff>�@l(�C�aH                                    Bxr�0  T          @�R@���=q@(Q�A���C�{@���
=>��H@z�HC�G�                                    Bxr��  {          @�
=@��\���@+�A��RC��@��\��=q?:�H@���C���                                    Bxr�|  
�          @�  @�{��{@8��A�{C��@�{��
=?c�
@��C���                                    Bxr�"  T          @�Q�@������H@1G�A�p�C�p�@������?:�H@�  C�XR                                    Bxr��  {          @��@����G�@7
=A��C��)@����=q?n{@�33C���                                    Bxrn  
�          @�@�����@C�
A�ffC���@�����?��A��C�,�                                    Bxr  �          @�ff@�(����@C33A��C�Ff@�(����?�{A{C��\                                    Bxr+�  �          @���@�z���Q�@=p�A��
C��=@�z����?s33@�
=C�S3                                    Bxr:`  
�          @�G�@�  ���@+�A���C��3@�  ���R>�@r�\C�                                      BxrI  T          @��@�\)���@+�A�G�C�� @�\)���R>�@tz�C�                                    BxrW�  �          @�Q�@�
=��(�@\)A���C���@�
=��ff>�z�@  C�
=                                    BxrfR  �          @�  @�=q��
=@p�A���C�P�@�=q���>�?�  C��                                    Bxrt�  T          @�@��\��=q@5�A��\C�K�@��\��=q?Q�@љ�C�#�                                    Bxr��  
�          @�@����ff@.�RA�\)C�` @������?.{@�\)C�j=                                    Bxr�D  -          @��H@\)��ff@
=A�  C�}q@\)��p��#�
���
C�p�                                    Bxr��  �          @��
@p  ����@z�A���C�9�@p  ���H�#�
���HC�P�                                    Bxr��            @�(�@fff���@(�A��
C�]q@fff�������%C��                                     Bxr�6  �          @�z�@e�����@  A��
C�L�@e���ff��=q�C���                                    Bxr��  
(          @��
@dz���ff@�RA�G�C�|)@dz���ff�#�
��Q�C�~�                                    Bxrۂ  
�          @��
@g
=���@�\A�33C���@g
=����B�\��G�C��3                                    Bxr�(  
e          @���@y����33@�A�p�C��)@y�����þ����C���                                    Bxr��  T          @���@|�����\@��A�  C�\@|����  �#�
��p�C�&f                                    Bxrt  �          @���@vff���@\)A�G�C��=@vff��=q�B�\��  C���                                    Bxr  "          @�@}p����@��A��
C�L�@}p���p���G��W
=C�Y�                                    Bxr$�  "          @�33@�
=���@0��A��RC�@�
=���\?aG�@��C��q                                    Bxr3f  
�          @��H@��
����@>{A��HC��\@��
���?��HA\)C��\                                    BxrB  �          @�G�@�����
=@2�\A�=qC�0�@�����ff?aG�@�p�C��                                    BxrP�  �          @��@��H��{@.{A�{C��@��H��(�?:�H@���C���                                    Bxr_X  "          @���@��
����@\)A�  C�J=@��
��33>Ǯ@E�C��f                                    Bxrm�  "          @�G�@��H���
@�A��C��3@��H���>��R@=qC���                                    Bxr|�  �          @陚@�(�����@#�
A���C��\@�(���z�?�@�ffC�{                                    Bxr�J  �          @��@��R��{@(Q�A��\C�L�@��R���H?(��@�ffC�t{                                    Bxr��  �          @��H@��H��33@'
=A�
=C�z�@��H��
=?
=@�(�C��q                                    Bxr��  "          @�=q@��H��=q@'�A�{C��@��H��ff?�R@��\C��=                                    Bxr�<  �          @��@�(����H@p�A�\)C��q@�(����>�@eC�f                                    Bxr��  	�          @�\@�=q����@�RA�{C�Y�@�=q����>�\)@
=qC��{                                    BxrԈ  
�          @���@�{��  @��A��C���@�{���>���@J=qC��                                    Bxr�.  �          @��@����@33A�C�޸@���=q>�{@*=qC�k�                                    Bxr��  �          @陚@�����@33A��HC��@����=q>�{@*�HC���                                    Bxr z  "          @�=q@�\)����@(�A��
C�  @�\)���H>��H@tz�C��                                    Bxr   �          @�\@������H@ ��A�(�C���@�����?�@�z�C��                                    Bxr�            @�G�@�z�����@=p�A�p�C�\@�z���G�?xQ�@���C��R                                    Bxr,l  T          @�Q�@������@@��A�(�C��
@������?��
A=qC��3                                    Bxr;  ,          @�=q@����z�@B�\A��C��3@����?�{A(�C��\                                    BxrI�  "          @��@�\)���\@/\)A�=qC��R@�\)��\)?+�@���C��                                    BxrX^  "          @��@�{���@7
=A��C���@�{��Q�?J=q@�ffC��3                                    Bxrg  �          @�G�@�p����
@,��A�ffC�k�@�p���  ?!G�@�\)C��f                                    Bxru�  �          @��@�\)��=q@1G�A�  C���@�\)��\)?5@��HC��                                    Bxr�P  
�          @��@�����\@.{A��C���@����\)?+�@���C�
=                                    Bxr��  "          @陚@�(���@.{A�G�C�
@�(���33?Q�@�C�,�                                    Bxr��  	�          @��H@�����@1�A�(�C��q@�����?fff@�\C���                                    Bxr�B  �          @�=q@�����\)@+�A��
C�f@�����(�?E�@���C�,�                                    Bxr��  |          @�\@�G����R@{A��C�s3@�G���G�?z�@��C���                                    Bxr͎  �          @��H@����=q@��A�ffC�H@�����>�@p  C�z�                                    Bxr�4  "          @�\@������@3�
A�ffC���@������H?W
=@љ�C��                                     Bxr��  "          @陚@����p�@1�A�\)C���@����33?O\)@�(�C��                                     Bxr��  	�          @�G�@�{��@'
=A��C���@�{��G�?&ff@�33C�!H                                    Bxr&  �          @��@��R���H@+�A��HC��@��R��\)?@  @��C�W
                                    Bxr�            @�Q�@�ff��@!G�A��RC�K�@�ff����?+�@�G�C��{                                    Bxr%r  ,          @�G�@��H��G�@&ffA�
=C��f@��H����?333@���C��                                    Bxr4  "          @�G�@����ff@1�A��C�S3@�����
?Q�@�{C���                                    BxrB�  �          @��@��H��
=@1�A��RC�Z�@��H��z�?O\)@�z�C���                                    BxrQd  T          @陚@�p����
@2�\A�C���@�p���G�?\(�@�G�C�
=                                    Bxr`
  "          @�G�@�G���Q�@0��A�{C��
@�G���?aG�@�{C��3                                    Bxrn�  "          @�G�@��\��
=@-p�A���C�Ф@��\��(�?Y��@�{C��{                                    Bxr}V  �          @�G�@�(���(�@2�\A�ffC�C�@�(���=q?u@�33C�E                                    Bxr��  �          @���@������\@$z�A��C�h�@�����?.{@�(�C��                                     Bxr��  �          @��@�Q����H@%�A�Q�C�H�@�Q���{?333@�  C��q                                    Bxr�H  T          @���@�G���@�A�=qC��@�G���ff>��@n{C���                                    Bxr��  
�          @���@�
=����@�A�C���@�
=����>�{@-p�C�=q                                    BxrƔ  �          @�G�@�=q����@�HA���C�B�@�=q��{?�@�ffC��f                                    Bxr�:  "          @陚@��R��Q�@��A���C��q@��R����?�@�(�C�'�                                    Bxr��  �          @�G�@����\)@)��A��HC�e@����33?:�H@�G�C���                                    Bxr�  
�          @��@�=q���@%A���C�9�@�=q���\?+�@��HC���                                    Bxr,  
�          @�Q�@�(����@ ��A�  C�k�@�(����?(�@�G�C��                                    Bxr�  �          @�  @��H��@p�A�C�8R@��H��z�>�{@.�RC���                                    Bxrx  �          @���@�(���{@
=qA�Q�C�T{@�(���z�>���@Q�C��                                    Bxr-  T          @�  @�G����@
=qA��HC��f@�G���>�z�@�C���                                    Bxr;�  �          @�  @�������@!�A��
C��@�����33?!G�@��RC��                                    BxrJj  "          @�  @�33���@/\)A��C�/\@�33���?L��@ʏ\C���                                    BxrY  �          @�\)@�Q�����@/\)A���C���@�Q�����?J=q@ə�C�5�                                    Bxrg�  
�          @�{@����=q@(��A�=qC�Z�@����p�?8Q�@���C��=                                    Bxrv\  "          @�\)@�G����@�A��RC��\@�G���33>�(�@XQ�C���                                    Bxr�  T          @�
=@�\)���@*�HA��C��R@�\)��33?J=q@�  C�T{                                    Bxr��  
Z          @�{@�p�����@%�A�{C��R@�p����
?.{@��RC��                                    Bxr�N  �          @�ff@����=q@(�A���C��f@����33?\)@�{C�^�                                    Bxr��  �          @�
=@�=q��Q�@�RA�z�C�.@�=q���?(�@�33C��R                                    Bxr��  T          @�
=@�����Q�@!G�A�C�!H@�����=q?(��@�  C���                                    Bxr�@  T          @�\)@������R@,��A��C�*=@������\?W
=@�C�~�                                    Bxr��  �          @�R@�����@'�A��RC��H@���z�?@  @�ffC��                                    Bxr�            @�\)@�=q��=q@�\A��C��=@�=q����>�{@,��C�b�                                    Bxr�2  �          @�  @�  ���R@
=qA��\C��R@�  ���
>8Q�?�33C���                                    Bxr�  T          @�
=@��
��Q�@�
A�G�C��q@��
��\)>Ǯ@FffC��=                                    Bxr~  �          @�@�z�����@�\A�G�C��@�z����>�p�@9��C��
                                    Bxr&$  T          @�\)@�ff��
=@G�A��RC�5�@�ff��{>\@?\)C�f                                    Bxr4�  T          @�
=@�{��ff@�
A��C�9�@�{��>�
=@W�C�                                      BxrCp  "          @�\)@�{��\)@  A�\)C�,�@�{��>�Q�@7�C��                                    BxrR  �          @�R@�����@
=qA��C��@���{>�=q@��C��\                                    Bxr`�  �          @�R@��H��Q�@�A���C���@��H��\)>�(�@[�C��\                                    Bxrob  �          @�R@�����
=@33A��RC�{@�����>�
=@U�C��H                                    Bxr~  
�          @�Q�@�{���H@A�C��@�{��\)>8Q�?��HC��                                    Bxr��  
�          @�  @�����@{A�Q�C��R@���\)>��
@#�
C��q                                    Bxr�T  
Z          @�
=@���z�@p�A�\)C�\)@���p�?(�@��\C���                                    Bxr��  �          @�@�����33@8Q�A���C�z�@�������?���AQ�C���                                    Bxr��  
�          @�\)@�(����@(��A�p�C�J=@�(���ff?L��@ʏ\C��                                    Bxr�F  �          @�\)@�����
@�HA�Q�C��q@����(�?
=@���C�Ff                                    Bxr��  T          @�R@�{��p�@33A��C�Q�@�{��z�>�@k�C��                                    Bxr�  �          @�@�\)���@
=qA�
=C�=q@�\)���>��R@��C�*=                                    Bxr�8  �          @�  @�ff��  @`��A�C�@�ff��z�?�  A_�
C�Y�                                    Bxr�  �          @�  @�  ��(�@9��A��RC�XR@�  ���?��AQ�C�}q                                    Bxr�  �          @�
=@�����Q�@#�
A�Q�C�
=@�����=q?G�@�{C���                                    Bxr*  "          @�@����(�@�HA�ffC���@����z�?(�@��HC�H�                                    Bxr-�  
�          @���@����=q@#�
A�ffC���@�����
?B�\@��C���                                    Bxr<v  �          @�  @�=q���H@�
A���C�� @�=q����>�G�@`  C�Y�                                    BxrK  �          @�  @�{��Q�@  A��HC�3@�{��ff>�
=@S33C��\                                    BxrY�  �          @�Q�@��H���@A���C���@��H���H?��@�(�C���                                    Bxrhh  T          @�\)@����\)@%A�C�=q@������?Tz�@ӅC��{                                    Bxrw  �          @�  @�  ����@��A��C��@�  ����>�@i��C��=                                    Bxr��  |          @�@|����z�@\)A�(�C�}q@|�����>���@C���                                    Bxr�Z  ,          @�Q�@w
=���\?�(�A{\)C�� @w
=���ͼ��
���C��                                    Bxr�   "          @�  @n{��
=?��AdQ�C��3@n{��\)�k���C�h�                                    Bxr��  �          @�R@W�����?ٙ�AY��C�XR@W����
��p��<��C��                                    Bxr�L  �          @�R@W
=��{?�ffAF�RC�AH@W
=�˅�����C��                                    Bxr��  �          @�ff@I����33?�
=A
=C�33@I����z�k����HC��                                    Bxrݘ  �          @�ff@@����z�?�\)A/�C�� @@���Ϯ�@  ��ffC�s3                                    Bxr�>  �          @�p�@L(����
?���A{�
C��H@L(���p���G��^�RC�33                                    Bxr��  �          @���@U�����?�AnffC�k�@U���=q�B�\��p�C��                                    Bxr	�  �          @�@tz���p�@�A��C�  @tz�����>W
=?�
=C�&f                                    Bxr0  �          @�{@w����@
�HA�ffC�+�@w�����>��@z�C�G�                                    Bxr&�  �          @�{@��\��
=@
=A�33C�Ф@��\��ff?z�@�(�C��R                                    Bxr5|  T          @�{@�����33@(�A�{C�P�@�������>Ǯ@FffC�E                                    BxrD"  
�          @�R@���
=@	��A��\C���@����
>��R@p�C���                                    BxrR�  T          @�ff@�����H@G�A�  C�  @����{>��?���C�/\                                    Bxran  T          @�{@�G���G�@�\A��RC���@�G����?�\@��\C�e                                    Bxrp  "          @�@���Q�@%A��C�� @���=q?aG�@�=qC�B�                                    Bxr~�  �          @�{@�����R@'�A���C��@������?n{@�ffC���                                    Bxr�`  
�          @�{@����G�@&ffA��
C��
@����33?c�
@���C��                                    Bxr�  
�          @�ff@�����@&ffA�\)C��=@�����
?c�
@�33C��                                    Bxr��  �          @�{@�\)��Q�@"�\A��
C��@�\)����?Y��@ڏ\C�p�                                    Bxr�R  T          @�R@�����@�A�p�C�=q@�����?333@��\C���                                    Bxr��  T          @�{@����33@{A��C��@������?�@�C��)                                    Bxr֞  �          @�{@�{��33@�A�Q�C���@�{��33?:�H@�33C�5�                                    Bxr�D  "          @�{@�\)���@A��C��q@�\)���H?#�
@��
C�]q                                    Bxr��  �          @�R@�33����@ffA�ffC�7
@�33��Q�?.{@�{C��                                    Bxr�  T          @�@������@  A��
C���@�����ff?.{@���C��=                                    Bxr6  T          @��@�ff��(�@z�A�p�C�t{@�ff���
?J=q@ə�C��                                    Bxr�  "          @�(�@�Q����R@(�A�Q�C���@�Q���\)?aG�@��C�&f                                    Bxr.�  T          @�{@�{���\@\)A�Q�C��3@�{���H?O\)@�Q�C�4{                                    Bxr=(  J          @��@�=q��(�@\)A��\C�@�=q����?J=q@��
C��R                                    BxrK�  
           @�{@�33��(�@!G�A��\C�"�@�33���?Tz�@�p�C��                                    BxrZt  |          @�{@������
@\)A�C�W
@�����(�?L��@�C���                                    Bxri  
�          @�p�@����\)@!�A�33C��@������?c�
@�z�C��3                                    Bxrw�  �          @��@�\)���@"�\A��HC��\@�\)��{?Y��@��
C�S3                                    Bxr�f  
�          @�@�������@*�HA�
=C���@������?p��@��C�@                                     Bxr�  �          @�p�@�ff����@(�A��C��\@�ff����?L��@���C�XR                                    Bxr��  �          @�@�(�����@(��A��RC�w
@�(����?}p�@�ffC���                                    Bxr�X  |          @�@�����\@,(�A��
C�(�@������?��
A�
C���                                    Bxr��  ,          @��@|����@�RA�Q�C��@|����p�?:�H@��C��)                                    BxrϤ  �          @�p�@�G���z�@#�
A��C��f@�G���?fff@�{C��f                                    Bxr�J  �          @�p�@�  ���
@*=qA�(�C���@�  ��?�  @��C�b�                                    Bxr��  �          @�@������H@!G�A�(�C�g�@������?aG�@�  C�
=                                    Bxr��  
Z          @�z�@�Q����\@�
A��\C�Q�@�Q���ff>�p�@@  C�b�                                    Bxr 
<  �          @��@�z���G�@&ffA�=qC���@�z����?�\)AQ�C��                                     Bxr �  �          @�p�@��
����@+�A�C��3@��
���?��HA�C��                                    Bxr '�  �          @�p�@��\����@(��A�33C�� @��\��z�?��RA�
C�
=                                    Bxr 6.  
(          @�R@��
����@,(�A��C�\@��
��z�?�ffA&=qC�,�                                    Bxr D�  
�          @�
=@��R���R@0��A���C�f@��R���\?��A'33C�,�                                    Bxr Sz  T          @�
=@�����(�@0��A���C�f@������?�G�A ��C�B�                                    Bxr b   
�          @�@�ff����@1G�A�{C�G�@�ff��(�?��RA��C���                                    Bxr p�  "          @�
=@��\��=q@3�
A�33C�Ff@��\��ff?�=qA*=qC�p�                                    Bxr l              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxr �              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxr ��   	          @�\)@�(���33@*=qA�ffC�ٚ@�(���{?�G�A ��C��                                    Bxr �^  �          @�\)@����(�@.{A�33C��3@����\)?���A(��C��R                                    Bxr �  T          @�@��H��{@&ffA�Q�C�n@��H��Q�?�
=A�\C��R                                    Bxr Ȫ  
�          @�\)@��
��\)@��A���C�j=@��
��  ?��A  C��
                                    Bxr �P  
�          @�@�z���{@   A��C��R@�z���\)?��A33C��R                                    Bxr ��  }          @�
=@������@!G�A�G�C�o\@�����?�z�A��C���                                    Bxr ��  
c          @�@��
��=q@(��A�p�C�k�@��
���?���A+�C���                                    Bxr!B  T          @�Q�@�����\)@'
=A���C��H@������?��\A ��C��{                                    Bxr!�  T          @�@�33���@'
=A�
=C�>�@�33��{?��A&ffC�e                                    Bxr! �  "          @�@�������@)��A���C�� @������
?�\)A.=qC���                                    Bxr!/4  �          @�@�����  @+�A�  C��)@�����33?�z�A4  C��f                                    Bxr!=�  
�          @�ff@��R���\@�RA���C���@��R��33?��A�C�)                                    Bxr!L�  "          @�\)@�����{@#33A��C��R@�����  ?��RA�C��                                    Bxr![&  �          @�@��R���@�RA�ffC��=@��R���?��HA�C��=                                    Bxr!i�  �          @�Q�@�����Q�@"�\A��C���@������?��HA��C���                                    Bxr!xr  �          @���@�{��z�@&ffA�G�C��q@�{��ff?�p�A�C�(�                                    Bxr!�  T          @�G�@������R@0  A�C�7
@������?�{A,Q�C�j=                                    Bxr!��  �          @�Q�@������@�A��C���@����
=?:�H@���C���                                    Bxr!�d  �          @�@�G���Q�@#33A�\)C���@�G���G�?��A
ffC�R                                    Bxr!�
  
�          @�Q�@�{��{@\)A�z�C�=q@�{���R?�ffA�C��                                     Bxr!��  "          @�  @�ff��{@(�A�G�C�C�@�ff��{?�  @��RC�Ф                                    Bxr!�V  �          @���@�  ��{@(�A���C�k�@�  ��{?�G�@��RC��R                                    Bxr!��  
�          @�@���G�@#�
A��C���@���33?��A&�HC���                                    Bxr!��  "          @���@�
=����@#�
A���C�@�
=���?��A&ffC���                                    Bxr!�H  T          @�Q�@�(��\)@2�\A���C��q@�(����?�ffAE��C��{                                    Bxr"
�  �          @�G�@�z���Q�@5A��C���@�z�����?���AJffC���                                    Bxr"�  
�          @�@�=q�vff@@��A�
=C�
=@�=q��G�?�Ag�C���                                    Bxr"(:  ]          @�  @�p��b�\@QG�AָRC�o\@�p����@	��A��
C��q                                    Bxr"6�  "          @�ff@�ff�=q@N�RA��
C���@�ff�N{@�A�  C��\                                    Bxr"E�            @�ff@�z��\(�@8Q�A�(�C�P�@�z����?�ffAg
=C��R                                    Bxr"T,  
�          @�{@�(��:=q@XQ�A���C�Y�@�(��n�R@�A��\C�7
                                    Bxr"b�  �          @�{@��
��ff@9��A��C�Ǯ@��
���\?��AEC��                                     Bxr"qx  
�          @�R@�G���{@=p�A�z�C��@�G���33?�
=AW�C��                                    Bxr"�            @�ff@�G����@'�A��HC�h�@�G���
=?��A%�C��
                                    Bxr"��  �          @�\)@��\���R@:�HA���C�*=@��\��33?У�AQ�C�#�                                    Bxr"�j  �          @�ff@�
=�x��@A�A�C��f@�
=���\?�Al��C�^�                                    Bxr"�  
�          @��@�ff�4z�@K�A�z�C�ٚ@�ff�e@G�A�33C���                                    Bxr"��  
�          @���@�\)��p�@W�A�33C�ٚ@�\)�5�@,(�A��C�S3                                    Bxr"�\  +          @���@�G���  @Y��A��HC��@�G��'�@1�A�33C�J=                                    Bxr"�  �          @�(�@����G�@N�RA��C��H@����5�@"�\A�33C�p�                                    Bxr"�  T          @��
@��
=@H��A�C�.@��HQ�@�A��C�R                                    Bxr"�N  �          @�@�33���
@L��A�  C�� @�33�%@%A���C��                                     Bxr#�  
�          @�(�@�p�����@j=qA��HC���@�p�� ��@EA�33C��H                                    Bxr#�  �          @���@�(���@G
=AυC���@�(��'�@\)A�C�h�                                    Bxr#!@  �          @�ff@�=q�A�@=p�A�z�C�>�@�=q�n{@G�A�ffC��\                                    Bxr#/�  T          @�ff@���A�@7�A�(�C�U�@���l��?�Q�Ay��C��                                     Bxr#>�  �          @�
=@���W
=@*=qA�33C���@���}p�?�33AR�HC���                                    Bxr#M2  �          @�{@����XQ�@'�A���C�ٚ@����}p�?���AMp�C�Ǯ                                    Bxr#[�  �          @�ff@����`  @��A��RC�n@�����G�?�33A333C��3                                    Bxr#j~  �          @�R@���]p�@�RA��C��)@����Q�?���A9p�C���                                    Bxr#y$  T          @�p�@����^�R@��A���C�xR@�����  ?�\)A/�
C��f                                    Bxr#��  �          @�p�@��R�G�@p�A�z�C�4{@��R�j�H?\AD(�C�5�                                    Bxr#�p  }          @�R@����'�@>{A£�C�>�@����U�@	��A���C���                                    Bxr#�  +          @�  @Å��R@?\)A�G�C���@Å�L(�@p�A�Q�C�5�                                    Bxr#��  �          @��@�����\@R�\A�p�C���@����E@#33A�z�C�}q                                    Bxr#�b  �          @�Q�@��
�$z�@;�A�ffC��)@��
�P��@Q�A�ffC��
                                    Bxr#�  �          @���@��
���@G
=AʸRC�K�@��
�I��@ffA���C�c�                                    Bxr#߮  �          @�G�@�=q�J=q@8Q�A��C��)@�=q�tz�?�Aw
=C�\)                                    Bxr#�T  �          @�\@�z��Vff@0  A�ffC�,�@�z��~{?�  A\��C�f                                    Bxr#��  �          @�@�{�`  @$z�A�\)C�@�{���?��A@��C���                                    Bxr$�  T          @��@���aG�@\)A��C���@�����?��HA8z�C��                                     Bxr$F  �          @陚@����mp�@(�A���C��H@�����\)?�{A,Q�C���                                    Bxr$(�  �          @��@����Q�@�\A�ffC�\)@����
=?��A�
C��R                                    Bxr$7�  �          @�=q@�G���ff@�A�G�C�c�@�G����?�=qA�C��{                                    Bxr$F8  �          @��@�=q���H@\)A�p�C���@�=q��G�?���A33C�k�                                    Bxr$T�  �          @��@��H���@�A�G�C�Ф@��H��G�?�  @�{C�o\                                    Bxr$c�  �          @���@��\��{@A���C��@��\��33?fff@�z�C�=q                                    Bxr$r*  }          @��@����Q�@�A�{C���@����p�?p��@��
C�0�                                    Bxr$��  +          @��@����  ?���Ax(�C�p�@�����?B�\@�
=C�J=                                    Bxr$�v  �          @�G�@������?��Ap(�C�޸@������R?+�@�Q�C��                                    Bxr$�  �          @���@�ff���?޸RA]C�>�@�ff��G�?   @y��C�T{                                    Bxr$��  �          @�\)@�ff�W�@8Q�A�{C��{@�ff��Q�?��Ar�RC�p�                                    Bxr$�h  T          @�p�@�\)�P  @4z�A�G�C�4{@�\)�xQ�?�{Ap��C��                                    Bxr$�  
�          @�{@�{�a�@&ffA�  C��@�{���H?˅AL  C�+�                                    Bxr$ش  �          @�@�Q��L��@333A�z�C�u�@�Q��u�?�\)Aq��C�1�                                    Bxr$�Z  �          @�R@�(��Y��@8��A��
C�p�@�(���G�?�33Au�C�,�                                    Bxr$�   }          @��@����e@!�A���C���@�����(�?�  AAp�C���                                    Bxr%�  �          @�z�@����_\)@G�A��C�y�@����~{?��\A$(�C��{                                    Bxr%L  T          @�p�@����e@�\A���C��@������?��\A#�C�o\                                    Bxr%!�  }          @�{@���g
=@{A��C�\@����=q?���A�C��                                     Bxr%0�  +          @��@��
�S33@��A���C�S3@��
�s�
?��HA<(�C��                                     Bxr%?>  �          @�R@�  �r�\@Q�A�33C�P�@�  ���R?���A(�C��                                    Bxr%M�  �          @�{@���p  @
=qA��C�n@����?�{AffC��
                                    Bxr%\�  �          @��@��\�dz�@Q�A�Q�C�>�@��\��  ?���A��C�                                    Bxr%k0  �          @���@�=q�^{@��A�p�C���@�=q�{�?��
A%p�C��)                                    Bxr%y�  �          @���@�33�XQ�@�
A���C��q@�33�w
=?�{A/
=C�G�                                    Bxr%�|  �          @��
@��H�AG�@,��A��\C�Y�@��H�g
=?�=qAn=qC�#�                                    Bxr%�"  �          @��
@����3�
@B�\A�z�C��@����`  @p�A���C�e                                    Bxr%��  �          @�(�@�{��
@K�A��C�Z�@�{�C�
@{A�ffC�b�                                    Bxr%�n  �          @�z�@�����@\(�A�ffC��\@����O\)@,��A��C�W
                                    Bxr%�  �          @��@�(��@VffA�p�C�%@�(��G�@(��A���C�                                      Bxr%Ѻ  �          @�(�@���!G�@J=qAң�C�]q@���P  @��A�C��                                     Bxr%�`  
�          @�z�@��H�\)@P  A�33C�q�@��H�O\)@   A�z�C�xR                                    Bxr%�  �          @�z�@����
@QG�A��C�P�@���Dz�@#�
A�ffC�B�                                    Bxr%��  T          @��
@�33�@R�\Aܣ�C�{@�33�G
=@%�A�=qC�                                      Bxr&R  T          @���@����#33@S�
A�G�C��@����S�
@#33A�p�C�R                                    Bxr&�  �          @�(�@�
=�AG�@>�RA��
C�3@�
=�l(�@
=A���C��)                                    Bxr&)�  �          @�(�@���0  @L��A��
C�33@���^�R@��A��C�e                                    Bxr&8D  �          @��@�Q��.{@R�\Aڣ�C�]q@�Q��^{@\)A�z�C�xR                                    Bxr&F�  T          @�z�@���.�R@X��A�\C�%@���`  @%�A�p�C�#�                                    Bxr&U�  "          @���@��
�'�@c�
A���C�}q@��
�\(�@1G�A��RC�@                                     Bxr&d6  �          @��
@����8Q�@Mp�A���C�~�@����fff@�A��C��q                                    Bxr&r�  T          @��
@�(��@��@J=qA���C��@�(��mp�@�\A�Q�C�H�                                    Bxr&��  �          @�@����8Q�@N{A�p�C�|)@����fff@Q�A���C��R                                    Bxr&�(  �          @�33@����4z�@O\)A�p�C���@����c�
@�HA�p�C��=                                    Bxr&��  �          @�@���5@S�
A��C��R@���e�@�RA�p�C���                                    Bxr&�t  T          @�33@��\�6ff@U�A��
C�u�@��\�fff@   A���C��3                                    Bxr&�  T          @�33@����J�H@E�A�(�C�"�@����u@�A�\)C���                                    Bxr&��  �          @��H@�=q�Mp�@>�RA���C�  @�=q�w
=@z�A�C��                                     Bxr&�f  �          @��H@�=q�U�@7
=A���C���@�=q�|(�?�
=A|��C�Y�                                    Bxr&�  T          @��H@�z��O\)@4z�A�G�C�\@�z��vff?�z�AyC���                                    Bxr&��  �          @��H@�33�W�@.{A��HC�}q@�33�|��?��Ai��C�k�                                    Bxr'X  T          @��H@�=q�`  @(Q�A�  C��{@�=q��G�?�z�AX��C�                                    Bxr'�  �          @�(�@�=q�^�R@1G�A���C�H@�=q��=q?�Aj�HC��                                    Bxr'"�  �          @�33@�(��e�@<��A�  C�33@�(���ff?���A33C��                                    Bxr'1J  �          @�(�@�z��j=q@Q�AۅC�G�@�z����
@  A�C���                                    Bxr'?�  �          @�z�@���x��@Q�A���C��@�����H@��A��C���                                    Bxr'N�  }          @�{@��|��@A�A�p�C�Q�@����H?�Q�Az�RC�<)                                    Bxr']<  �          @��@��\����@?\)A��
C���@��\��?��At  C��\                                    Bxr'k�  T          @�p�@����(�@7�A�ffC���@����
=?޸RA`��C���                                    Bxr'z�  �          @��
@����|(�@,(�A�  C���@�����\)?�\)AS
=C��                                    Bxr'�.  �          @�z�@�
=�h��@0  A���C�.@�
=���R?�  Ac�
C�4{                                    Bxr'��  T          @�R@�����
@�\A�{C��@������?�\)A�C���                                    Bxr'�z  �          @�R@�33����@\)A��C��
@�33��{?�{A��C�aH                                    Bxr'�   �          @�ff@��s33@.{A���C���@����?�Q�AZ{C��H                                    Bxr'��  �          @�R@�z��Dz�@Q�A�z�C���@�z��r�\@=qA�ffC�\                                    Bxr'�l  �          @��@�\)�q�@:�HA�G�C��
@�\)��(�?��Ap��C���                                    Bxr'�  �          @�@��R�z�H@+�A��
C�,�@��R���R?�\)AO
=C�aH                                    Bxr'�  �          @�  @�G���\)@�RA��C��3@�G����R?�{A-p�C�*=                                    Bxr'�^  �          @�@�����
@33A�z�C�G�@�����?fff@�C�!H                                    Bxr(  T          @�\)@�
=��\)@ffA�p�C�%@�
=���
?}p�@��
C���                                    Bxr(�  T          @�
=@�  ���@A���C��)@�  ���
?h��@��C���                                    Bxr(*P  �          @�\)@�  ���
@ffA��C�q@�  ��=q?���AQ�C��)                                    Bxr(8�  �          @�
=@�����@z�A�\)C�)@����\)?^�R@�z�C��                                    Bxr(G�  T          @�
=@���k�@,(�A���C�\)@����\)?�Q�AX��C�|)                                    Bxr(VB  �          @�ff@�33�qG�@�RA���C��@�33����?�p�A=G�C�Z�                                    Bxr(d�  �          @�
=@��H�w
=@�A��C��\@��H���H?�33A333C��                                    Bxr(s�  "          @�@���qG�@'
=A��\C��@������?˅AK33C�C�                                    Bxr(�4  �          @�\)@�G��_\)@$z�A��HC�s3@�G�����?У�AP��C��q                                    Bxr(��  T          @�@����Y��@.�RA�\)C��@����}p�?�Ag�C���                                    Bxr(��  �          @���@���N�R@.�RA���C���@���s33?���AlQ�C��                                     Bxr(�&  
�          @�  @���B�\@-p�A��C���@���g
=?��ApQ�C�t{                                    Bxr(��  �          @�  @�(��L��@1�A��RC��3@�(��r�\?�z�Atz�C���                                    Bxr(�r  �          @���@���R�\@,��A�(�C�w
@���vff?�Af=qC�w
                                    Bxr(�  �          @���@�  �B�\@333A�z�C��3@�  �hQ�?��HAz{C�h�                                    Bxr(�  �          @��@�  �A�@1G�A�p�C��)@�  �g�?�Q�AxQ�C�u�                                    Bxr(�d  �          @��@�ff�J=q@0��A�(�C��@�ff�o\)?�33Aq�C���                                    Bxr)
  �          @�G�@�  �?\)@5A��C��H@�  �fff@G�A���C���                                    Bxr)�  �          @��@�  �@��@9��A��C��\@�  �hQ�@�A��C�k�                                    Bxr)#V  �          @��@�G��c�
@-p�A��C�0�@�G����
?�  A]��C�H�                                    Bxr)1�  �          @��@�p��tz�@#�
A�(�C��@�p����\?��AC\)C�O\                                    Bxr)@�  �          @���@���~{@   A���C�@ @�����R?��HA8��C���                                    Bxr)OH  "          @���@�{�fff@.{A�{C�ٚ@�{���?�G�A`(�C��\                                    Bxr)]�  
�          @�  @���tz�@(��A�
=C��f@����33?У�AO�C�                                      Bxr)l�  T          @��@����@!�A���C�s3@���33?�Q�A7�C�޸                                    Bxr){:  �          @��@�Q��y��@(Q�A�ffC�Y�@�Q���?���AK�
C��)                                    Bxr)��  T          @�Q�@��
���H@)��A���C�\)@��
���?���AHQ�C���                                    Bxr)��  T          @�G�@�  �g�@)��A�=qC��H@�  ���?�
=AUG�C��                                    Bxr)�,  �          @�Q�@����j�H@��A�ffC���@�������?�
=A5��C�'�                                    Bxr)��  �          @�  @�{�O\)@%�A���C��3@�{�qG�?��HAZ�RC���                                    Bxr)�x  �          @�\)@�G��b�\@\)A��HC�H�@�G���G�?�ffAEC��                                    Bxr)�  �          @�  @���y��@��A�
=C��3@����(�?�
=A4Q�C��                                    Bxr)��  T          @�  @�33�z=q@�HA��C��@�33��(�?��A1G�C���                                    Bxr)�j  T          @�@��H�qG�@(Q�A��\C���@��H����?У�APz�C�33                                    Bxr)�  
�          @�  @�z��p  @%�A���C�33@�z���Q�?˅AK
=C�t{                                    Bxr*�  �          @�  @�z��h��@-p�A��C���@�z���{?޸RA^�HC��                                    Bxr*\  �          @�@�p��k�@&ffA�ffC��f@�p���ff?У�AO�
C��q                                    Bxr*+  �          @�Q�@�  �e�@(��A�(�C�f@�  ���
?�
=AVffC�/\                                    Bxr*9�  �          @��@�
=�e�@*�HA���C��)@�
=���
?�(�A[33C�q                                    Bxr*HN  �          @�@�G��aG�@!�A���C�W
@�G�����?˅AK�C��3                                    Bxr*V�  ~          @�
=@�=q�^�R@   A�C��\@�=q�~�R?�=qAIC��\                                    Bxr*e�  �          @���@��H�`��@#�
A�
=C�~�@��H����?У�AO�C��3                                    Bxr*t@  T          @�G�@�{�W�@!�A�ffC�9�@�{�xQ�?У�AO33C�l�                                    Bxr*��  �          @��@��U�@$z�A��C�U�@��w
=?�
=AVffC�z�                                    Bxr*��  �          @�  @����Tz�@&ffA�ffC�P�@����vff?�(�A[�C�l�                                    Bxr*�2  T          @�@��QG�@#�
A��C���@��r�\?�Q�AW�C��{                                    Bxr*��  �          @��@��U@"�\A�\)C�P�@��vff?�33AQ�C�~�                                    Bxr*�~  �          @陚@���@  @C33A�p�C���@���i��@�RA�Q�C�+�                                    Bxr*�$  �          @�=q@����E@L��A��C��R@����qG�@
=A��\C�w
                                    Bxr*��  �          @�\@��H�Z�H@4z�A�
=C��3@��H��  ?�z�AqG�C��=                                    Bxr*�p  �          @��H@�33�U�@;�A�(�C�.@�33�|(�@�A��RC�                                    Bxr*�  T          @�33@��
�X��@3�
A�{C���@��
�~{?�33Ap��C���                                    Bxr+�            @�=q@�G��N{@%�A�
=C���@�G��p  ?�(�AY��C��                                    Bxr+b  �          @陚@�G��Mp�@#33A��C��@�G��n�R?ٙ�AW33C�&f                                    Bxr+$  �          @�G�@�Q��N�R@%�A��C�� @�Q��p��?�(�AZ{C��q                                    Bxr+2�  �          @��@�ff�U�@(��A��C�` @�ff�w�?�  A]��C�y�                                    Bxr+AT  T          @�=q@���S33@(��A���C���@���u?�G�A^ffC���                                    Bxr+O�  �          @���@�33�X��@,��A�  C��{@�33�|(�?�ffAdz�C��                                    Bxr+^�  "          @��@�p��N{@.{A��C��R@�p��r�\?�{AmG�C��3                                    Bxr+mF  �          @陚@�\)�QG�@'
=A�  C��f@�\)�s�
?޸RA]G�C��                                     Bxr+{�            @�G�@���XQ�@.{A�\)C�f@���|(�?���Ag�C�\                                    Bxr+��  �          @�G�@��H�W
=@2�\A�{C��@��H�{�?��Ap��C��                                    Bxr+�8  
�          @�Q�@���\��@333A�33C�~�@������?��Ao�C�z�                                    Bxr+��  
�          @�Q�@��\�XQ�@-p�A�G�C��
@��\�{�?�Ag
=C�                                      Bxr+��  �          @��@�=q�Z�H@,(�A���C���@�=q�~{?��
Ab�RC��)                                    Bxr+�*  T          @��@���[�@&ffA��C�Ф@���}p�?�Q�AW
=C��R                                    Bxr+��            @��@����\(�@   A��C���@����|(�?���AJ�HC�!H                                    Bxr+�v  �          @���@�{�XQ�@   A��RC�/\@�{�xQ�?�{AL  C�j=                                    Bxr+�  T          @�G�@��
�^{@#�
A���C��
@��
�~�R?�33AP��C��=                                    Bxr+��            @���@����_\)@-p�A���C�s3@�����G�?��
Ab�HC���                                    Bxr,h  �          @陚@��H�W�@1G�A��RC���@��H�|(�?�\)Am�C��)                                    Bxr,  T          @�G�@����Z�H@1�A�G�C��)@����\)?�{AmG�C��)                                    Bxr,+�  T          @��@��\�Q�@;�A�
=C�O\@��\�x��@�\A�  C�!H                                    Bxr,:Z  �          @陚@�\)�hQ�@+�A�=qC��{@�\)��p�?�(�AYC���                                    Bxr,I   T          @�G�@�33�W�@1G�A�ffC��@�33�|(�?�\)Amp�C�                                      Bxr,W�  �          @�G�@��,(�@2�\A�{C�<)@��R�\@�
A���C��q                                    Bxr,fL  T          @���@�(���\@*�HA�  C�"�@�(��7�@�A�C��f                                    Bxr,t�  
�          @陚@��#�
@9��A���C���@��K�@(�A�(�C�`                                     Bxr,��  �          @���@�G��@?\)A�(�C���@�G��0  @��A�C�5�                                    Bxr,�>  �          @���@�Q��Q�@G�A�C�h�@�Q��(��@#�
A��C���                                    Bxr,��  
�          @�G�@����)��@7�A��C�\)@����P��@��A��HC�                                    Bxr,��  T          @���@���g
=@(��A�ffC��@����z�?�Q�AV�\C�
                                    Bxr,�0  �          @�G�@����c33@(��A�=qC�9�@������H?ٙ�AX(�C�`                                     Bxr,��  T          @�G�@�(��[�@'�A�=qC��)@�(��~{?��HAXQ�C�                                      Bxr,�|  �          @���@�{�P��@*�HA��C���@�{�s�
?�ffAdQ�C���                                    Bxr,�"  
�          @���@�G��7
=@7�A�(�C�T{@�G��^{@A�C��                                    Bxr,��  �          @��@�\)�8Q�@<��A�C�"�@�\)�aG�@
=qA�z�C���                                    Bxr-n  T          @���@�{�HQ�@2�\A��RC�R@�{�n{?���Axz�C���                                    Bxr-  T          @�G�@���S�
@1G�A��HC�B�@���x��?��Ao�C�9�                                    Bxr-$�  T          @陚@�G��X��@8��A��RC�Ф@�G��\)?�p�A{�
C���                                    Bxr-3`  �          @��@�p��A�@AG�A�33C�o\@�p��j�H@(�A��C�{                                    Bxr-B  T          @�=q@�z��W�@0  A�z�C�  @�z��{�?���Ai�C�!H                                    Bxr-P�  �          @�\@����n{@!G�A��\C���@�����
=?��AAp�C��q                                    Bxr-_R  T          @�33@���l(�@p�A�Q�C��@����?��RA:ffC�AH                                    Bxr-m�  �          @�33@�G��n{@%�A�{C���@�G���\)?˅AH(�C���                                    Bxr-|�  �          @��H@��~�R@Q�A���C�z�@���{?��A(Q�C���                                    Bxr-�D  �          @�\@�{�tz�@%�A�G�C�{@�{���\?�=qAF�RC�]q                                    Bxr-��  �          @�=q@���}p�@
=A�C��@����p�?���A)�C���                                    Bxr-��  �          @��@�
=�e@2�\A�C���@�
=��p�?�=qAhQ�C���                                    Bxr-�6  �          @�=q@�p��p��@,(�A�z�C�8R@�p�����?�Q�AUp�C�ff                                    Bxr-��  
�          @�=q@�(��vff@)��A���C��\@�(���(�?�\)AMG�C�                                    Bxr-Ԃ  �          @���@�33�y��@#33A�C���@�33����?\A@Q�C��                                    Bxr-�(  T          @��
@��\��Q�@)��A���C�(�@��\��G�?˅AG\)C�u�                                    Bxr-��  �          @�33@�(��u�@1�A��C��q@�(�����?�G�A]�C�                                      Bxr. t  �          @��
@�ff����@�A�(�C���@�ff����?��
A��C�Q�                                    Bxr.  �          @�z�@�z����?ǮAC33C�  @�z���G�>�{@'
=C�p�                                    Bxr.�  �          @�(�@�z����\?���A(z�C��@�z�����=�?p��C��                                    Bxr.,f  �          @���@�����(�@  A��C�Ǯ@�������?�{A
=qC�~�                                    Bxr.;  �          @��@�����H@��A�G�C��{@������?�G�A��C�l�                                    Bxr.I�  �          @���@�����@
=qA��
C���@����z�?�ffA�HC�O\                                    Bxr.XX  �          @�z�@�
=��(�@A�  C��=@�
=��=q?���AG�C�O\                                    Bxr.f�  �          @�(�@�33���R@�
A�  C��=@�33����?��HA�\C�(�                                    Bxr.u�  �          @�p�@�ff��33@Q�A���C�&f@�ff���?�ffA!�C���                                    Bxr.�J  ~          @�ff@���z�@(�A��C��@����
?���A&�RC�q�                                    Bxr.��  �          @�p�@���p  @%A�\)C���@������?˅AE��C���                                    Bxr.��  �          @�z�@�Q쿳33@%A�ffC�!H@�Q��   @
=qA�ffC���                                    Bxr.�<  �          @�33@�p���@4z�A�z�C���@�p��333@(�A�
=C�33                                    Bxr.��  �          @��
@�p��
=@:�HA�33C���@�p��1G�@�
A�z�C�U�                                    Bxr.͈  �          @�33@����H@<��A��C���@��'�@�A�
=C��                                    Bxr.�.  �          @�\@�ff���@*=qA�  C�Y�@�ff�5@G�A�C��                                    Bxr.��  �          @��H@�ff�#�
@=p�A�(�C��=@�ff�Mp�@�RA�C�Q�                                    Bxr.�z  �          @�@�p��QG�@9��A�C��f@�p��x��@   A|z�C�XR                                    Bxr/   �          @��
@��R�qG�@,(�A�  C�C�@��R��=q?�
=AS
=C�n                                    Bxr/�  �          @��
@�{����@Q�A�33C���@�{���H?���A	�C�ff                                    Bxr/%l  �          @�@�������@�A�Q�C��
@������?�R@�(�C�                                      Bxr/4  �          @���@����(�?��A`(�C�z�@����z�>�\)@��C��                                    Bxr/B�  T          @��
@��\�_\)@1G�A�G�C��f@��\��=q?�Ae�C��f                                    Bxr/Q^  �          @�(�@�(��5@:�HA���C���@�(��^�R@�A�(�C�8R                                    Bxr/`  �          @�33@�
=�>{@EA�
=C��=@�
=�h��@��A��C�O\                                    Bxr/n�  �          @�@�=q�:=q@>{A�ffC�4{@�=q�c33@	��A�{C��3                                    Bxr/}P  �          @�@��
�(Q�@G
=A�Q�C�\)@��
�Tz�@
=A�C���                                    Bxr/��  �          @�@�  �=q@B�\AÙ�C�o\@�  �Fff@ffA��HC�Ф                                    Bxr/��  �          @��
@�
=�=q@HQ�A�33C�k�@�
=�G
=@�A�=qC���                                    Bxr/�B  �          @�@�ff� ��@C�
A��HC��)@�ff�L(�@A�z�C�^�                                    Bxr/��  �          @�\@����(��@J�HA�33C�5�@����Vff@=qA�C��                                    Bxr/Ǝ  �          @��H@��
�'
=@EA�p�C�q�@��
�S33@A���C��{                                    Bxr/�4  �          @��
@�=q��R@7�A�33C�K�@�=q�G�@
=qA�(�C��                                     Bxr/��  �          @��
@ʏ\�p�@5A�C�b�@ʏ\�E@��A�
=C���                                    Bxr/�  �          @�\@�G��<(�@8Q�A��C��@�G��dz�@33A�ffC��{                                    Bxr0&  �          @��H@�=q�W�@;�A�(�C���@�=q�\)?��RA{�C��                                    Bxr0�  �          @�(�@����<��@@  A�z�C��@����g
=@
�HA��HC���                                    Bxr0r  T          @��@��W�@8Q�A��C�0�@��\)?�Q�As33C��                                    Bxr0-  �          @���@�G�����@333A��C���@�G���33?�Q�AS�
C�%                                    Bxr0;�  �          @�z�@����{�@:=qA�G�C�Z�@�������?�=qAd��C�`                                     Bxr0Jd  �          @�p�@�ff��Q�@AG�A�{C���@�ff��z�?�z�An�RC��=                                    Bxr0Y
  T          @��@����z�@9��A�z�C�G�@�����?�G�A\Q�C�c�                                    Bxr0g�  T          @���@������@7�A���C�aH@�������?�Q�AS�C��{                                    Bxr0vV  �          @��@�\)��{@1G�A��
C���@�\)��  ?ǮAB�\C�(�                                    Bxr0��            @�{@������@/\)A��HC��
@����=q?�G�A;
=C��
                                    Bxr0��  �          @�p�@����@-p�A�C�aH@����\)?���AC�C��                                    Bxr0�H  �          @�p�@�G����H@/\)A��C��=@�G�����?�\)AIG�C�H                                    Bxr0��  �          @�{@�33��=q@,(�A�  C��)@�33���
?���AC�C�:�                                    Bxr0��  T          @�@�{��33@{A�
=C�q@�{���H?���A&�HC���                                    Bxr0�:  �          @�z�@�  ����@(��A��C�xR@�  ��{?�  A<  C��f                                    Bxr0��  �          @��
@��R��z�@0��A�=qC��\@��R��ff?�ffAB�RC�<)                                    Bxr0�  �          @�(�@�����@�HA���C�9�@����Q�?�{A
�RC���                                    Bxr0�,  �          @�(�@������H@��A�33C��@�������?�=qAffC��=                                    Bxr1�  �          @�(�@�{���@(�A�=qC��f@�{���H?�  A\)C�.                                    Bxr1x  �          @��@�����=q@(�A��
C���@�����G�?�Q�A(�C�8R                                    Bxr1&  �          @���@�  ���@
=A�ffC�0�@�  ���?��A�HC��H                                    Bxr14�  �          @�(�@��
��\)@z�A���C�
@��
��?���A	�C���                                    Bxr1Cj  �          @�z�@��H��?��RAz{C�` @��H����?8Q�@��C�W
                                    Bxr1R  �          @��@��R���R?�Aa�C�4{@��R��  >�@k�C�^�                                    Bxr1`�  �          @�{@�=q��p�?�Q�AQC�9�@�=q��>���@�C��f                                    Bxr1o\  �          @�p�@�G���ff?У�AJ�HC��@�G���ff>u?�=qC�`                                     Bxr1~  �          @�p�@�=q��z�?ٙ�AT  C�N@�=q����>��
@{C��
                                    Bxr1��  �          @��@�����R?�33AMC���@�����R>�  ?�
=C�9�                                    Bxr1�N  �          @�{@����G�?���As
=C�aH@����z�?5@��C�U�                                    Bxr1��  �          @�R@����(�@   AzffC�!H@����  ?L��@�
=C���                                    Bxr1��  �          @�ff@�33���R?�  A9G�C���@�33��>#�
?��RC��3                                    Bxr1�@  �          @�{@�������?�  A:ffC�*=@�����  >��?�33C��\                                    Bxr1��  T          @�@�p���  ?��A^�HC�c�@�p�����?   @u�C���                                    Bxr1�  �          @�p�@�{��33@�
A���C�� @�{��\)?J=q@���C�                                    Bxr1�2  �          @��@����G�?�Q�A2�HC���@����\)<#�
=���C�&f                                    Bxr2�  �          @��@�z����
?�(�A6�RC�*=@�z���=q<��
>��C��f                                    Bxr2~  �          @�p�@��\��ff?�z�A/�
C��H@��\��(��L�;�
=C�J=                                    Bxr2$  �          @�{@��
��Q�?u@�C�aH@��
��녿����C�AH                                    Bxr2-�  �          @�R@�33���?E�@��RC�.@�33��녿E���C�,�                                    Bxr2<p  �          @�@����z�?�z�AN�\C��q@����(�>B�\?��HC�<)                                    Bxr2K  �          @�@�ff��{?�z�Ao33C��3@�ff��Q�>��H@q�C��R                                    Bxr2Y�  �          @�p�@�p�����?�{AHQ�C�k�@�p���G�>#�
?��RC���                                    Bxr2hb  T          @�p�@�{����?�33AMG�C��=@�{����>L��?���C���                                    Bxr2w  �          @�p�@�
=��  ?��A#\)C��@�
=��z�u���C��
                                    Bxr2��  �          @���@�\)��\)?���AG�C���@�\)�����
=q���C��                                     Bxr2�T  �          @���@�33����?�(�AY��C��\@�33��{?   @|(�C��                                    Bxr2��  �          @���@�
=��Q�?�p�AXQ�C��@�
=���\?(�@�\)C��                                    Bxr2��  �          @�@�p��vff@��A��C��H@�p�����?�z�A  C�                                      Bxr2�F  �          @�p�@������?�G�A=G�C�E@����Q�>B�\?�G�C���                                    Bxr2��  �          @�@�33��z�?���AG�C�^�@�33��(�>8Q�?�z�C��
                                    Bxr2ݒ  �          @���@����
=?�=qAdz�C�y�@������?�\@~{C���                                    Bxr2�8  �          @�@�����\)?�
=Ap��C�c�@������\?(�@�C�e                                    Bxr2��  �          @�@�G���=q?�(�Au��C��{@�G���p�?(�@�\)C��{                                    Bxr3	�  �          @�{@�Q�����@ffA�\)C�K�@�Q���p�?Q�@��HC�)                                    Bxr3*  �          @�ff@�ff��33@Q�A�
=C��\@�ff��  ?Tz�@�(�C���                                    Bxr3&�  �          @�R@�(���@A�=qC��f@�(����\?Tz�@��C���                                    Bxr35v  �          @�R@�=q���@ffA��C��\@�=q��z�?Tz�@��
C�\)                                    Bxr3D  "          @�ff@�p����
@z�A�\)C�4{@�p���Q�?Tz�@���C��q                                    Bxr3R�  �          @�Q�@�G���  @�A���C��)@�G���?u@�33C���                                    Bxr3ah  �          @�G�@��
��
=@Q�A�G�C�S3@��
��{?��\@��C��)                                    Bxr3p  �          @�G�@�=q��ff@(�A�
=C���@�=q��ff?�33A��C�5�                                    Bxr3~�  �          @��@�������@\)A���C�o\@�����G�?�  A��C�ٚ                                    Bxr3�Z  �          @�ff@�G��w
=@'
=A�=qC�'�@�G���p�?�(�A6ffC�S3                                    Bxr3�   �          @�@���tz�@'�A�\)C�S3@����z�?��RA9��C�w
                                    Bxr3��            @�p�@���j�H@+�A�G�C��{@����Q�?�=qAEG�C��q                                    Bxr3�L  )          @�(�@�z��e@(Q�A���C�W
@�z����?ǮAC33C�aH                                    Bxr3��  �          @�z�@�Q��aG�@��A�33C�Ф@�Q����?�z�A/�C���                                    Bxr3֘  �          @�(�@����hQ�@$z�A���C�5�@�����{?��RA:=qC�O\                                    Bxr3�>  �          @�z�@�  �]p�@$z�A���C��@�  ����?��
A?�
C�\                                    Bxr3��  �          @�z�@�
=�tz�@,(�A���C�q@�
=���?�ffAAG�C�,�                                    Bxr4�  �          @���@��\�qG�@%�A���C���@��\���\?���A4��C��{                                    Bxr40  �          @��
@�ff�{�@!�A��RC���@�ff��
=?�{A*ffC���                                    Bxr4�  �          @�\@������@ ��A��\C��R@�����\?��A$��C�AH                                    Bxr4.|  �          @��H@�=q���H@\)A�Q�C��)@�=q���?��\A�C�.                                    Bxr4="  �          @�@�
=��p�@Q�A�p�C���@�
=����?���A�C�{                                    Bxr4K�  �          @�p�@��H���H@�
A��C��@��H����?xQ�@�\)C�Q�                                    Bxr4Zn            @�@�ff��\)@ffA�G�C�N@�ff��ff?��
@�C��                                    Bxr4i  �          @�\)@�33��\)@{A�=qC�E@�33���?Tz�@�z�C��                                    Bxr4w�  �          @�
=@�=q����@(�A��\C�\@�=q��{?J=q@\C�ٚ                                    Bxr4�`  �          @�ff@����{@(�A���C�n@�����?O\)@�\)C�1�                                    Bxr4�  �          @�
=@�����
=?�A`Q�C���@�����Q�>�z�@��C�+�                                    Bxr4��  �          @�
=@�33����@�A|z�C�=q@�33���>�@l��C�P�                                    Bxr4�R  T          @�ff@�����\)@A��
C���@�����33?��@��C��
                                    Bxr4��  T          @�33@�G���ff@   A��HC���@�G���
=?�z�A�C�aH                                    Bxr4Ϟ  �          @���@�����@�RA��C�` @�����H?W
=@�Q�C��                                    Bxr4�D  �          @��
@�����?�33Ao�C���@������>�@qG�C��                                     Bxr4��  �          @陚@�������@{A���C���@�����
=?\(�@ٙ�C�W
                                    Bxr4��  �          @�=q@��H��
=@\)A���C��@��H���?fff@�=qC���                                    Bxr5
6  T          @�\@�33��Q�@�A��\C���@�33��{?Tz�@�  C��H                                    Bxr5�  �          @��@������@p�A��\C�q�@�����H?aG�@�{C�
                                    Bxr5'�  �          @�G�@�����@A��C�Q�@������H?B�\@�
=C�3                                    Bxr56(  �          @�G�@�{��z�@z�A�  C���@�{��G�?@  @��C�N                                    Bxr5D�  �          @陚@�ff��(�@ffA��C��{@�ff����?G�@�(�C�L�                                    Bxr5St  �          @�=q@�p����@�\A��C�*=@�p���z�?0��@���C��R                                    Bxr5b  �          @�z�@��
���H@A�C��
@��
��  ?333@�
=C���                                    Bxr5p�  �          @�{@�(����\?�\A\Q�C��f@�(����
>��?�p�C�ٚ                                    Bxr5f  �          @�p�@�����33?��Ak33C�Z�@�����>�Q�@0  C�y�                                    Bxr5�  �          @��@�33���?���Ac�C��R@�33���
>��R@�C���                                    Bxr5��  �          @��@������?���Ag
=C��@�����R>���@C�4{                                    Bxr5�X  �          @���@�z�����?��HAUC�j=@�z�����>�?��\C���                                    Bxr5��  �          @�z�@�����?�Q�AS�C��)@����p�=�\)?�C��3                                    Bxr5Ȥ  �          @�z�@�ff���?��
A_�C�}q@�ff��ff>#�
?�p�C��                                    Bxr5�J  �          @�z�@�ff���?�G�A\��C�~�@�ff��{>�?��C��=                                    Bxr5��  �          @��@�G�����?��ALQ�C��=@�G�����<��
=�C�*=                                    Bxr5��  �          @���@��H����?�{Ai��C�E@��H���H>�=q@z�C�s3                                    Bxr6<  �          @�p�@�=q��=q?���Ag�
C�)@�=q��(�>u?��C�N                                    Bxr6�  �          @�p�@��\��z�?�z�AN�HC��=@��\��z�<�>��C�E                                    Bxr6 �  �          @�p�@�z����\?У�AK�C�E@�z���=q<��
>.{C���                                    Bxr6/.  �          @�(�@��H���?ٙ�ATz�C�.@��H��=q=��
?(��C�}q                                    Bxr6=�  �          @�(�@�  ��{?��A'\)C�'�@�  ���H��p��8Q�C��\                                    Bxr6Lz  T          @�(�@������\?�AQ�C��)@������H=L��>\C�Q�                                    Bxr6[   �          @�(�@�������?�(�Aw\)C�3@������>�33@,(�C�.                                    Bxr6i�  �          @�@�ff��(�?�  A\��C���@�ff���=�G�?W
=C���                                    Bxr6xl  �          @�(�@�33��Q�?�ffAb{C�Z�@�33����>B�\?���C��3                                    Bxr6�  �          @�z�@�z���G�?��
A?
=C���@�z�����#�
��  C�xR                                    Bxr6��  �          @���@�p���(�?�
=Ar=qC���@�p���  >��H@q�C���                                    Bxr6�^  �          @�Q�@�  ����@ ��A�z�C��q@�  ��ff?(��@�ffC��q                                    Bxr6�  �          @�  @�����
=@z�A�z�C�P�@�����z�?:�H@��C��)                                    Bxr6��  �          @�G�@�33���
@
�HA��HC���@�33���\?\(�@ٙ�C�\)                                    Bxr6�P  �          @�  @�33���H@Q�A�ffC���@�33����?Q�@У�C�~�                                    Bxr6��  T          @�  @�(���  @��A��C�K�@�(���\)?k�@陚C�                                    Bxr6�  �          @�\)@�=q�}p�@
=A�z�C�J=@�=q��\)?�=qA	��C���                                    Bxr6�B  �          @�R@�33��Q�@�A���C�1�@�33���R?W
=@��C��)                                    Bxr7
�  �          @�R@�����  @�A���C�XR@�����p�?@  @��C��
                                    Bxr7�  �          @�
=@��|(�@z�A�\)C���@���(�?L��@�z�C�/\                                    Bxr7(4  �          @�
=@�p��|��@ffA�\)C���@�p�����?Tz�@�=qC��                                    Bxr76�  �          @�\)@�z���G�@33A�p�C�7
@�z����R?@  @�p�C��{                                    Bxr7E�  �          @�\)@�ff�y��@
=A��C��=@�ff���?W
=@ָRC�L�                                    Bxr7T&  �          @�  @���y��@
=A��C��f@����33?Y��@�
=C�g�                                    Bxr7b�  �          @�@��}p�@
=A�p�C���@����?Q�@�Q�C�
                                    Bxr7qr  �          @�@�
=�z=q@A�{C���@�
=���?O\)@�{C�U�                                    Bxr7�  �          @�\)@����q�@�
A��\C�q�@�����\)?Tz�@��C��{                                    Bxr7��  �          @�\)@�33��=q@�A���C���@�33��  ?5@��
C��H                                    Bxr7�d  �          @�ff@����
?�ffAh(�C��3@����R>���@J=qC��                                    Bxr7�
  �          @�@�p���{?�33As�
C��f@�p���G�>Ǯ@FffC�ٚ                                    Bxr7��  �          @��@�{��z�@Q�A�=qC��H@�{���?
=q@�G�C�u�                                    Bxr7�V  �          @�G�@�=q����@�A�z�C�E@�=q���R?�@�=qC�q                                    Bxr7��  �          @��@�G����@�
A��C�&f@�G����R>��H@xQ�C��                                    Bxr7�  �          @�@�{��  ?�(�A[�C�S3@�{����=�G�?\(�C���                                    Bxr7�H  �          @�
=@�z���
=?��HA\  C��R@�z�����>8Q�?��HC�ٚ                                    Bxr8�  �          @�@�
=��(�?�=qAiC���@�
=��\)>���@I��C��
                                    Bxr8�  �          @�
=@�z����R?�G�Ab�\C�,�@�z���G�>��R@p�C�.                                    Bxr8!:  �          @�\)@�����?У�AP��C���@����ff>L��?���C���                                    Bxr8/�  �          @�\)@�Q����?�AUp�C�=q@�Q���ff>��?�Q�C�b�                                    Bxr8>�  �          @�
=@��\��=q?�33AS�
C���@��\���>#�
?��
C��3                                    Bxr8M,  �          @�
=@������H?��HA[\)C���@�������>W
=?�33C���                                    Bxr8[�  �          @�\)@�Q���Q�@33A�C��3@�Q���p�?��@�(�C�t{                                    Bxr8jx  �          @�\)@��R��33?�p�A~�RC�Ff@��R���>�@hQ�C��                                    Bxr8y  �          @�R@��
����@\)A���C�1�@��
����?5@�C���                                    Bxr8��  �          @���@����@�A�=qC�N@�����?@  @�z�C��q                                    Bxr8�j  �          @��@��R���H@
=A�\)C�S3@��R����?z�@��C��                                    Bxr8�  �          @�\)@�{����@��A�{C���@�{��\)?��@�  C���                                    Bxr8��  �          @�\)@�{��z�@�A�33C�� @�{��?n{@��C�9�                                    Bxr8�\  �          @�@�\)����@
�HA���C���@�\)��
=?&ff@��C�8R                                    Bxr8�  �          @�\)@�ff��z�@G�A��RC�ff@�ff��z�?Y��@ٙ�C�˅                                    Bxr8ߨ  �          @�  @�����@�RA�C�j=@���  ?0��@��C�                                      Bxr8�N  �          @�  @��H��p�@�A�(�C���@��H���
?��@�\)C�h�                                    Bxr8��  �          @�Q�@�p���{@   A�  C�o\@�p���=q>�{@*=qC�aH                                    Bxr9�  �          @��@�=q���
@   A~{C��@�=q��  >�Q�@2�\C��)                                    Bxr9@  �          @���@�����{@A��C��{@������?   @z=qC���                                    Bxr9(�  �          @�Q�@�Q����@�
A�(�C��=@�Q���\)?�\@���C�H�                                    Bxr97�  �          @��@�Q����@A�C��f@�Q����?�@�{C�@                                     Bxr9F2  �          @��@�G�����@G�A��HC���@�G����R>�@hQ�C�q�                                    Bxr9T�  �          @�G�@�\)����@�A��C�1�@�\)����>�G�@_\)C�                                      Bxr9c~  �          @�Q�@��R��p�?�33As
=C�\@��R����>���@=qC���                                    Bxr9r$  T          @���@�  ��?�\)Am��C�!H@�  ����>�=q@C��                                    Bxr9��  �          @��@�G����?�z�As\)C�q�@�G���\)>���@#�
C�Y�                                    Bxr9�p  �          @�@�����{?�33As
=C�N@������>�p�@:�HC�'�                                    Bxr9�  �          @�  @�������?�Q�Axz�C���@�����>�p�@<(�C��=                                    Bxr9��  �          @�@������\?�33As33C���@�����ff>��
@   C�h�                                    Bxr9�b  �          @�  @�����?�Av{C�|)@�����>Ǯ@G�C�L�                                    Bxr9�  �          @�@�(���(�@�
A�Q�C�k�@�(����?�@�ffC�
                                    Bxr9خ  �          @�R@�����\@ffA�C��q@����G�?+�@���C�aH                                    Bxr9�T  �          @�
=@��
�~�R@
=qA�p�C�\)@��
���R?@  @�
=C��f                                    Bxr9��  �          @�@�ff�|(�@�A�ffC���@�ff��p�?8Q�@��C�)                                    Bxr:�  �          @�@�\)�s33@  A�33C�=q@�\)���\?c�
@�=qC�|)                                    Bxr:F  �          @�\)@����k�@33A�Q�C���@�����\)?xQ�@�C��H                                    Bxr:!�  �          @�{@���tz�@\)A�
=C�H@�����H?\(�@�z�C�E                                    Bxr:0�  �          @�ff@�=q��  @�A���C�,�@�=q���?=p�@�p�C��3                                    Bxr:?8  �          @�
=@��H����@�A�Q�C��@��H����?+�@�G�C���                                    Bxr:M�  �          @�R@��
�~�R@	��A�
=C�XR@��
��
=?8Q�@�  C��H                                    Bxr:\�  �          @�R@����n�R@
=qA��C���@������?O\)@�C�޸                                    Bxr:k*  �          @�{@�33�c33@{A�  C�Y�@�33���H?k�@�C���                                    Bxr:y�  �          @�p�@�ff�P��@�
A�=qC��H@�ff�u?���A��C���                                    Bxr:�v  �          @�R@\�G�@�\A�(�C�n@\�l��?�\)A�C�W
                                    Bxr:�  �          @�  @����
=@�HA�z�C���@����B�\?�  A?33C�N                                    Bxr:��  �          @�Q�@�{�%�@1G�A��C���@�{�W
=?�G�A_�
C��f                                    Bxr:�h  �          @�R@�p��(��@'�A��\C�g�@�p��W
=?˅AK�C���                                    Bxr:�  �          @�R@ƸR�>{@
�HA�=qC�7
@ƸR�aG�?��A�\C�33                                    Bxr:Ѵ  �          @�ff@�Q��:�H@33A�Q�C�|)@�Q��\(�?s33@�C���                                    Bxr:�Z  Z          @�R@�z��:�H@�A�  C�B�@�z��c33?�  A ��C��R                                    Bxr:�   �          @�@��H�C�
@��A��RC��@��H�l(�?�(�A  C�c�                                    Bxr:��  �          @��@����?��RA~�\C�/\@���*=q?�z�A\)C��                                    Bxr;L  �          @�@љ���
@G�A�p�C�K�@љ��6ff?�\)AffC�<)                                    Bxr;�  �          @�
=@�Q����?��RA�=qC��q@�Q��<(�?��A�C��)                                    Bxr;)�  �          @�ff@�
=�!�?�33At��C�S3@�
=�AG�?n{@�p�C�z�                                    Bxr;8>  �          @�
=@��H�3�
@G�A��C�
=@��H�U�?p��@�C�&f                                    Bxr;F�  �          @�\)@�G��:=q@33A�{C���@�G��\(�?p��@�\)C��=                                    Bxr;U�  �          @�ff@�\)�6ff@p�A���C��
@�\)�[�?�{Ap�C��{                                    Bxr;d0  �          @�{@��
�G
=@
=A���C���@��
�i��?k�@���C���                                    Bxr;r�  �          @�ff@Ǯ�7
=@z�A�=qC���@Ǯ�Y��?xQ�@�Q�C���                                    Bxr;�|  �          @�p�@����aG�?���Am��C��
@����{�?�\@��\C�(�                                    Bxr;�"  �          @��@��R����?��HA;�C��\@��R��(��\)���C���                                    Bxr;��  �          @�p�@�  ���?��RA@Q�C���@�  ��33��Q�B�\C�f                                    Bxr;�n  �          @�@�p����?���A1��C�,�@�p���ff��  �   C��                                    Bxr;�  �          @�
=@�{��G�?�{A.{C��@�{��\)��z���\C��H                                    Bxr;ʺ  �          @�
=@����Q�?��\A"�RC�N@����p���Q��8Q�C��                                    Bxr;�`  �          @��@�����z�?��
A=qC��\@�����ff�:�H����C�g�                                    Bxr;�  �          @��@��\��=q?��A	�C��{@��\���Ϳ&ff��(�C���                                    Bxr;��  �          @���@����Q�?(��@��C���@����p����H�z�C���                                    Bxr<R  �          @�G�@�����R?\)@�p�C�
@�����\����"�HC�s3                                    Bxr<�  �          @�G�@�����=q?���A(�C�AH@�����ff����j�HC�ٚ                                    Bxr<"�  �          @陚@�����?�  @���C�S3@����\�:�H��\)C�,�                                    Bxr<1D  �          @�G�@�����?��A
=qC�� @����{�.{���HC���                                    Bxr<?�  �          @��@�p����?�z�A4��C��@�p����׾�\)�(�C�P�                                    Bxr<N�  �          @�G�@�z���Q�?���A�RC�G�@�z����\�.{���\C��                                    Bxr<]6  �          @�\@��R����?�ffA33C��\@��R��33�L����Q�C��\                                    Bxr<k�  �          @�=q@�
=��Q�?�\)A(�C��@�
=���\�8Q���p�C��                                    Bxr<z�  �          @�\@�����?�  Az�C���@����p���R��(�C�q�                                    Bxr<�(  �          @�\@�p����?���A6ffC��@�p�������N{C�z�                                    Bxr<��  �          @�G�@��R����?�33A0��C���@��R��녿���33C��                                    Bxr<�t  �          @���@�(���=q?\(�@���C�G�@�(����׿���	��C�e                                    Bxr<�  �          @��@�z�����?(��@�{C��3@�z���p���{�,��C�G�                                    Bxr<��  �          @���@�����R>�@g�C��@����  ��{�L(�C��)                                    Bxr<�f  �          @�  @��H��G�>��H@y��C�o\@��H���\��{�L��C��\                                    Bxr<�  �          @�Q�@������?#�
@��C��)@������H���H�9G�C��                                    Bxr<�  �          @���@�\)��{?s33@�  C���@�\)��p������RC���                                    Bxr<�X  �          @���@����G�?J=q@�
=C�@�����R��  ��RC�=q                                    Bxr=�  �          @��@�=q��=q?�{A��C��@�=q����aG���p�C��)                                    Bxr=�  �          @���@�����?��
AffC��\@����(��xQ���p�C��                                    Bxr=*J  T          @���@�(���G�?fff@�(�C��@�(������z���RC�\                                    Bxr=8�  �          @��@����?fff@�\C��=@����(���\)�p�C��f                                    Bxr=G�  �          @��@�����ff?s33@�ffC���@�����p�����	C���                                    Bxr=V<  �          @�\@�����  ?E�@���C��@�����p���Q����C�(�                                    Bxr=d�  �          @�\@�\)���?Tz�@�\)C���@�\)�����z��C��                                    Bxr=s�  �          @�G�@����\)?\(�@أ�C��{@����{��{��
C���                                    Bxr=�.  �          @陚@������?=p�@��C��@������Ϳ�(���\C�&f                                    Bxr=��  �          @��@������
?O\)@��HC�0�@�����G����H���C�e                                    Bxr=�z  �          @陚@�{����?u@�\C��@�{���׿����HC��R                                    Bxr=�   �          @陚@�{����?333@�
=C�
@�{��p�����/�C�n                                    Bxr=��  �          @�\@���=q?��\A ��C��@����H�h�����C���                                    Bxr=�l  �          @�33@�  ���R?�\)A,  C�L�@�  ���
�����=qC���                                    Bxr=�  �          @�33@�{��  ?��A/33C��@�{����������C��R                                    Bxr=�  �          @��@������?@  @���C�H@����G���ff�#�
C�H�                                    Bxr=�^  �          @��@�z����H?Tz�@У�C�>�@�z����׿��H���C�p�                                    Bxr>  �          @�=q@������?��\A z�C���@������0�����RC�Y�                                    Bxr>�  �          @�=q@�=q���?�=qA�C�AH@�=q��{�k���{C�+�                                    Bxr>#P  T          @��@�(����?�  @���C���@�(�����xQ����
C���                                    Bxr>1�  �          @�\@�z�����?�33A0��C��@�z���{�����C�Z�                                    Bxr>@�  �          @��@������\?�  A=��C�^�@������׿�\�~{C���                                    Bxr>OB  T          @陚@�����(�?�\)A,z�C�33@�����Q�&ff���C�Ф                                    Bxr>]�  �          @���@��H��z�?0��@��C���@��H��  �����/\)C�W
                                    Bxr>l�  �          @陚@��\���?���A
ffC��@��\��(��}p����\C���                                    Bxr>{4  �          @��@������?�p�A\)C�q�@������W
=���
C�=q                                    Bxr>��  �          @��@�p���  ?��A"�\C���@�p���33�G����C�L�                                    Bxr>��  �          @陚@��\��33?��A�
C�
=@��\��z�u���HC��3                                    Bxr>�&  �          @陚@��
��  ?��A)p�C�l�@��
���
�@  ��(�C�q                                    Bxr>��  �          @�G�@��H����@G�A��C�Y�@��H���R=u>�G�C�.                                    Bxr>�r  �          @�Q�@��
���?���A(z�C�c�@��
���R�Q���Q�C�"�                                    Bxr>�  �          @�Q�@�z��^{@>�RA�p�C�0�@�z����H?�
=A5C�4{                                    Bxr>�  �          @�Q�@���9��@VffA�z�C���@���}p�@   A�  C��3                                    Bxr>�d  �          @�Q�@�(��aG�@:=qA���C���@�(����
?���A+�C�
                                    Bxr>�
  �          @�G�@�����ff@   A���C�@������H?333@���C��                                    Bxr?�  T          @陚@������R@�A��C���@�������>�@s33C�8R                                    Bxr?V  T          @�G�@�Q���p�@�A�C��R@�Q�����?��@��C�.                                    Bxr?*�  �          @陚@����=q@"�\A���C�k�@����
=?.{@�33C�t{                                    Bxr?9�  �          @�G�@��H����@!G�A��C��\@��H���?.{@��
C���                                    Bxr?HH  �          @�Q�@���G�@(��A�ffC��=@����?aG�@�C�k�                                    Bxr?V�  �          @�  @�������@\)A��HC�7
@�����G�?333@�Q�C�33                                    Bxr?e�  �          @�Q�@�
=��z�@Q�A��C�s3@�
=��\)?
=@�ffC��\                                    Bxr?t:  �          @�Q�@��R���@�A�C��@��R��Q�>�Q�@7
=C�xR                                    Bxr?��  �          @���@������@
=A�\)C��3@������?��@�\)C�
=                                    Bxr?��  �          @��@��H����@��A��RC��@��H���?   @~{C�>�                                    Bxr?�,  T          @�  @�=q���H@
�HA�\)C��{@�=q���
>���@K�C�*=                                    Bxr?��  �          @���@�����ff@��A���C�b�@�����
=>Ǯ@B�\C��)                                    Bxr?�x  T          @���@�\)��p�?���Ax��C��@�\)��논��uC���                                    Bxr?�  �          @��@�33��p�?�\A`(�C�@�33��
=�����C�,�                                    Bxr?��  �          @�\@�{����?��AO�C��@�{��z��\�~{C�Ff                                    Bxr?�j  �          @�=q@��\��G�?���A)C�Ǯ@��\���
�p�����
C���                                    Bxr?�  T          @�\@�\)��{?���Ap�C�)@�\)��p������
=C�&f                                    Bxr@�  �          @��
@��
���?��HA33C���@��
��zῌ���	p�C���                                    Bxr@\  �          @�33@�{���?��A(��C�B�@�{���\�n{��G�C�
=                                    Bxr@$  �          @�@�Q���{?���A(z�C��R@�Q����ÿk����C�^�                                    Bxr@2�  �          @�@����p�?\A>=qC��R@����=q�@  ��(�C�0�                                    Bxr@AN  �          @�z�@��R��  ?�  A;�C�G�@��R��z�L���ǮC��=                                    Bxr@O�  �          @�\@�
=��ff?��A$��C�q�@�
=���׿u���C�B�                                    Bxr@^�  �          @�33@��R��ff?���A6{C�l�@��R���\�W
=���C��                                    Bxr@m@  �          @��H@����?�z�AQ��C�P�@����z�#�
��\)C��                                    Bxr@{�  �          @�\@������@   A|��C��)@�����  �����%�C�q                                    Bxr@��  '          @�\@�(���p�?�p�A[\)C���@�(������p��C��{                                    Bxr@�2  �          @�=q@�
=����?�\)Am�C�S3@�
=���������C�k�                                    Bxr@��  T          @��@�p����?�=qAHQ�C���@�p���녿(�����C�g�                                    Bxr@�~  �          @�Q�@�(���(�?�ffAE��C�˅@�(���=q�#�
���\C�Ff                                    Bxr@�$  T          @�  @������?�{A,z�C�aH@������J=q��G�C��                                    Bxr@��  �          @�@�z���=q?�33AS�C�g�@�z����\��(��Z=qC���                                    Bxr@�p  �          @��@�����p�?�\A`��C�Ff@�����\)��=q�Q�C�W
                                    Bxr@�  �          @�Q�@�Q���p�?�  A_�C�4{@�Q���
=��z���\C�K�                                    Bxr@��  �          @�R@�=q��Q�?�ffAh(�C���@�=q��33�8Q쿴z�C��\                                    BxrAb  �          @���@�33��\)?�33AU�C��
@�33�������QG�C�Ф                                    BxrA  �          @��@�z���(�?�{Ap  C��@�z���\)�B�\��  C��{                                    BxrA+�  T          @�@����
=?�z�Av�RC��H@�����H�8Q쿵C�n                                    BxrA:T  �          @�@�33���
@�\A�=qC��\@�33�������
�\)C���                                    BxrAH�  �          @�{@�����\)?�Axz�C�o\@�������8Q쿴z�C�Y�                                    BxrAW�  �          @�p�@�����ff?��HA}G�C���@������H�\)����C�h�                                    BxrAfF  �          @�@�33��33@�
A�33C��
@�33��G��#�
����C���                                    BxrAt�  �          @���@�����
@�A��
C��@����33>\)?���C���                                    BxrA��  �          @���@������@z�A��RC�(�@����  <��
>\)C�Ф                                    BxrA�8  �          @�p�@�z����?���A|��C��@�z���Q���p��C�޸                                    BxrA��  �          @�p�@�(����?�
=Az{C�
=@�(�����\)���C���                                    BxrA��  �          @���@�=q��z�?�Q�A|z�C��@�=q���þ����Q�C���                                    BxrA�*  �          @�p�@��H���R?�An{C��R@��H������=q��C��R                                    BxrA��  �          @�{@�(���ff?�ffAhQ�C��)@�(����þ��R�{C���                                    BxrA�v  �          @�(�@�=q��p�?�=qAm�C��=@�=q��Q쾊=q���C���                                    BxrA�  �          @�@�Q���  ?�Q�A[�C�C�@�Q����׾�G��c33C�xR                                    BxrA��  �          @�z�@�ff���\?�G�Ac�
C��R@�ff���
��
=�UC�H                                    BxrBh  �          @�p�@�  ���?��
Af{C��@�  ���
�Ǯ�G�C�+�                                    BxrB  T          @��@�  ���?�\Ad��C��@�  ��������Mp�C�5�                                    BxrB$�  �          @�@�
=��(�?�G�Ac�C���@�
=��p���G��^�RC��\                                    BxrB3Z  �          @�{@�����?�  AaC��@�����;�ff�c�
C��                                    BxrBB   �          @�p�@�������?�  AaC�+�@������H��(��Z�HC�T{                                    BxrBP�  �          @�p�@����=q?�{AP  C�1�@�������\)���C��                                    BxrB_L  �          @�(�@�\)��(�?\AD��C�Ǯ@�\)��=q�.{��{C�>�                                    BxrBm�  �          @�p�@��H���?�ffAG�C�Q�@��H��Q��R��\)C��R                                    BxrB|�  �          @�\)@�p����\?�p�A<Q�C�y�@�p���  �333���\C���                                    BxrB�>  �          @�
=@����?�Q�AXQ�C�q@����ff��ff�c33C�L�                                    BxrB��  �          @�\)@�����G�?ǮAF�HC���@������׿�\���\C�E                                    BxrB��  �          @�@�p���=q?�(�A<(�C��\@�p���Q�����Q�C�XR                                    BxrB�0  �          @�\)@��
���
?�z�A4  C���@��
���ÿ0����{C�*=                                    BxrB��  T          @�  @�����z�?�\)A.=qC���@������ÿ:�H���C�<)                                    BxrB�|  �          @�p�@������?�  A ��C�ff@������Y����G�C�                                      BxrB�"  �          @�R@�����p�?��
A�
C���@�����������HC���                                    BxrB��  �          @�R@�(����R?xQ�@�\)C�ff@�(���p��������C���                                    BxrC n  �          @�
=@�������?n{@�{C��@���������H��\C�
                                    BxrC  �          @�ff@�  ��33?n{@��C���@�  ���ÿ��R�=qC��
                                    BxrC�  �          @�R@��R���H?�Ap�C���@��R���
���\�{C�w
                                    BxrC,`  �          @�
=@�p�����?�A��C�J=@�p�����ff�G�C�4{                                    BxrC;  �          @�ff@�����z�?�33A�HC�<)@�������������C�/\                                    BxrCI�  �          @�{@�����
?��AffC�U�@����zῈ�����C�H�                                    BxrCXR  �          @�@�z����
?�z�Ap�C�B�@�z����Ϳ�ff��HC�/\                                    BxrCf�  �          @�
=@�ff��33?�Q�A(�C�z�@�ff��zῂ�\��\C�]q                                    BxrCu�  �          @�\)@������?�
=A6ffC�T{@�����  �Q��ϮC���                                    BxrC�D  �          @��@�  ��
=?޸RA]C��@�  ��Q���p��C�*=                                    BxrC��  �          @�\)@�\)��\)?���AI�C��@�\)��{�#�
���C�K�                                    BxrC��  �          @�
=@�����
?�G�A ��C�S3@����{�}p����C�#�                                    BxrC�6  �          @�
=@�p����\?�A5p�C�s3@�p���
=�Tz��ҏ\C�                                    BxrC��  �          @�\)@�ff����?�G�AA�C���@�ff���R�:�H��Q�C�.                                    BxrC͂  �          @�G�@�����=q?�Q�A5�C�˅@������R�O\)����C�`                                     BxrC�(  �          @�G�@�p���33?�AS�
C�j=@�p����H��R��z�C��R                                    BxrC��  �          @�G�@��H��\)@�A��C��@��H���R��G��fffC�/\                                    BxrC�t  �          @��@����z�@�A�Q�C��H@�����R=���?B�\C��                                    BxrD  �          @�Q�@�33��G�?���Ax��C�aH@�33�����33�333C�Q�                                    BxrD�  �          @�G�@�(���(�?޸RA\Q�C�1�@�(����Ϳ
=��(�C�o\                                    BxrD%f  �          @�\@�����\)?�\)Al��C��@�����=q�����J=qC�                                    BxrD4  �          @���@�ff����?��
Ac33C��)@�ff��=q�   �|(�C�޸                                    BxrDB�  �          @��@���Q�?�Ai�C��3@����H���eC���                                    BxrDQX  �          @���@����\)?�  A^=qC���@�����׿�����C�                                      BxrD_�  �          @���@�Q����?�ATQ�C���@�Q���������  C�@                                     BxrDn�  �          @���@������?\AA��C�)@�����p��=p�����C���                                    BxrD}J  �          @�G�@�������?�G�A@  C���@������^�R��z�C�&f                                    BxrD��  �          @��@�{���?��A)C�� @�{��
=�����
�RC��R                                    BxrD��  �          @��@�  ���\?�{A-p�C�Ff@�  ��������
C��                                    BxrD�<  �          @�G�@��
��33?�=qAh��C�E@��
��������
C�c�                                    BxrD��  �          @�Q�@�(�����?���Ag�C�q�@�(���������C��\                                    BxrDƈ  �          @�
=@��H����?�G�AbffC�` @��H��=q�\)���C��                                    BxrD�.  �          @�\)@��\��(�?�z�AS�C��@��\��33�333���\C�p�                                    BxrD��            @�{@�  ��\)?��A,��C���@�  �������
��C�Z�                                    BxrD�z  �          @�(�@����?aG�@��HC��\@�������33�W\)C�K�                                    BxrE   
�          @��@�����?z�H@���C�]q@����
�����O�C���                                    BxrE�  �          @�ff@K����L�;�
=C�"�@K���
=�<(����C���                                    BxrEl  �          @�
=@xQ�����?=p�@��
C�aH@xQ����R��
����C��                                    BxrE-  �          @�
=@J=q��\)�#�
��p�C���@J=q���R�C�
��
=C�y�                                    BxrE;�  �          @�{@-p�����=�\)?
=qC�
@-p����R�<(�����C�9�                                    BxrEJ^  �          @�
=@<����33=���?G�C�
=@<�����9����
=C�8R                                    BxrEY  �          @�\)@6ff���
>��H@z�HC���@6ff��(��#�
���C�xR                                    BxrEg�  �          @�Q�@:�H��(�>�ff@c33C��@:�H�Å�'
=��Q�C��=                                    BxrEvP  �          @�\)@.�R��{>8Q�?�Q�C�q@.�R��G��7����C�+�                                    BxrE��  �          @�
=@,(���ff�\)���C���@,(������J�H�У�C�=q                                    BxrE��  T          @�p�@  ��녽�G��k�C�!H@  ��Q��L����(�C�@                                     BxrE�B  �          @��@
=�׮��\)��C���@
=���
�Tz����
C��{                                    BxrE��  T          @���@&ff���>��?�C���@&ff����:=q��{C��
                                    BxrE��  �          @��@�
�أ�=���?W
=C�o\@�
��=q�@  ��ffC�o\                                    BxrE�4  �          @�p�@&ff��=��
?+�C���@&ff��\)�>�R�ģ�C�                                    BxrE��  �          @���@!�������ffC�Y�@!���(��K���G�C��                                     BxrE�  �          @��
@{��Q쾏\)�G�C�R@{���
�Vff���C�Y�                                    BxrE�&  �          @�?�G���(���\)���C�+�?�G���\)�Z=q��\)C�0�                                    BxrF�  �          @���?�ff��
=�aG���G�C�Ff?�ff���H�Y����\)C�&f                                    BxrFr  �          @�?�����
��  ��\C��{?������X����
=C���                                    BxrF&  �          @��H@��������ͿG�C���@������I����C�                                      BxrF4�  �          @��
@������R�\)C��)@������W
=��G�C�b�                                    BxrFCd  �          @�\@�
��{��G��k�C��@�
��(��L(���z�C��R                                    BxrFR
  �          @�\@8Q���
==�\)?z�C��@8Q������;���p�C�Ff                                    BxrF`�  �          @�  @G����\�G�C�l�@G�����[���33C��{                                    BxrFoV  �          @߮?�\��
=�333��
=C�c�?�\��(��n�R�=qC��q                                    BxrF}�  �          @�  ?��أ׿
=q���C��?�����g���p�C�q                                    BxrF��  �          @�?�Q���=q��Q��<(�C���?�Q����
�`  ���
C�&f                                    BxrF�H  �          @��H@����\)������HC�\@����z��QG��ۮC�AH                                    BxrF��  �          @��@
�H�����
�0��C��q@
�H���
�L(���  C�"�                                    BxrF��  �          @�33@#�
���
>�=q@	��C���@#�
��\)�6ff���C���                                    BxrF�:  �          @���@ff�Ӆ>�{@333C���@ff��  �1�����C��{                                    BxrF��  �          @�Q�@�
��33?�\@���C��=@�
��=q�(������C�o\                                    BxrF�  �          @�
=@����=q?
=q@�
=C�xR@������&ff��
=C�4{                                    BxrF�,  �          @�ff@��У�?
=@��HC���@������"�\��p�C��=                                    BxrG�  �          @��@!���G�>��@VffC��\@!����R�-p���p�C�y�                                    BxrGx  �          @��@+���
=>��
@'
=C�B�@+�����0����\)C�K�                                    BxrG  �          @��@<(���33=��
?!G�C�ff@<(���z��:�H��G�C���                                    BxrG-�  �          @��
@����(�?Tz�@�C�AH@�������G����HC���                                    BxrG<j  �          @�z�@���
=?c�
@�RC�O\@���(������p�C��=                                    BxrGK  �          @�{@\)�θR>�\)@�C���@\)���\�3�
��33C���                                    BxrGY�  �          @�33@8Q��θR��  ��\C�f@8Q���=q�QG���z�C���                                    BxrGh\  �          @�\@HQ��ə��B�\��p�C�33@HQ���{�i�����\C���                                    BxrGw  �          @��H@<����ff�.{��33C�G�@<�����H�Mp���
=C��                                    BxrG��  �          @�\@8���θR    =#�
C��@8����{�C�
���HC�|)                                    BxrG�N  �          @�=q@N{��Q�����Q�C���@N{����]p�����C��H                                    BxrG��  �          @��@Z�H���
�����P��C���@Z�H��{�P����G�C��                                    BxrG��  �          @�  @Z=q��=q�.{���HC��=@Z=q�����_\)��G�C�)                                    BxrG�@  �          @�\)@L(�������G�C���@L(���{�Y����C��=                                    BxrG��  T          @޸R@Mp����;���XQ�C���@Mp����R�R�\��Q�C���                                    BxrG݌  �          @�{@U��\��=q�  C�\)@U����R�H����p�C�J=                                    BxrG�2  �          @�p�@U��G���p��B�\C�xR@U��(��Mp���G�C���                                    BxrG��  �          @޸R@H�����\)���
C�xR@H������\����C��=                                    BxrH	~  �          @�\)@L(�������ffC���@L(������]p����
C��                                    BxrH$  �          @�Q�@=p���녿=p���=qC��3@=p����k�����C��H                                    BxrH&�  �          @�  @K���{�J=q��{C���@K������j=q��  C��                                    BxrH5p  �          @�
=@Vff��G��p����Q�C�~�@Vff��33�n�R�Q�C�P�                                    BxrHD  �          @ڏ\@(���Ǯ�Tz���Q�C�w
@(����=q�n�R��C��                                     BxrHR�  �          @�33@0���ƸR�W
=���C���@0����G��n�R�C�XR                                    BxrHab  �          @�33@-p���  �L����ffC���@-p����H�mp���C�H                                    BxrHp  �          @ڏ\@-p���ff�^�R��=qC�˅@-p���Q��p  ��C�0�                                    BxrH~�  �          @�=q@
�H����L���׮C�\)@
�H��\)�s33�ffC�H�                                    BxrH�T  �          @��H@(����G��љ�C�e@(���  �r�\��RC�N                                    BxrH��  T          @��H?�  ��z����Q�C�Z�?�  �����mp���\C���                                    BxrH��  �          @��?Ǯ���H�@  ��G�C���?Ǯ�����u�	p�C��                                    BxrH�F  �          @ٙ�?�  �Ӆ�   ��ffC�^�?�  ��G��h����\C���                                    BxrH��  �          @�33?�=q��
=�.{��
=C��)?�=q��G��Z=q���HC�z�                                    BxrH֒  �          @�=q?�=q���#�
����C���?�=q��=q�Q���{C�l�                                    BxrH�8  �          @ٙ�?��H��>��?�p�C�"�?��H�����G�����C��=                                    BxrH��  �          @�G�?��\�ָR���k�C�N?��\��33�R�\��{C���                                    BxrI�  �          @أ�?h����ff��  ���C�� ?h����
=�^�R��{C��f                                    BxrI*  �          @أ�?������
�.{��33C��f?�����ff�XQ���C��                                    BxrI�  �          @�G�?�Q���녾8Q�\C�=q?�Q���z��W
=��C�]q                                    BxrI.v  �          @��?��
��(���G��n{C���?��
��\)�U���HC�~�                                    BxrI=  �          @���?���녾�\)���C�#�?����H�]p���\)C�S3                                    BxrIK�  T          @أ�?�\�θR�fff��p�C���?�\��{�|���
=C�l�                                    BxrIZh  �          @أ�?��H��\)��G��,Q�C�Z�?��H���������  C��                                    BxrIi  �          @�Q�?�
=��Q쿎{��\C�,�?�
=��(������C�                                    BxrIw�  �          @�Q�?�G��љ�������C�o\?�G���p���p��Q�C���                                    BxrI�Z  �          @�Q�?�\)���þǮ�VffC���?�\)����c33���\C�5�                                    BxrI�   �          @�Q�?��H���ÿ8Q��ÅC�H�?��H���\�u�{C���                                    BxrI��  �          @׮?�\)��{�!G���(�C��?�\)��G��n{�(�C��
                                    BxrI�L  �          @�\)?�(������H�&ffC�n?�(���  ��\)�(�C�0�                                    BxrI��  �          @�Q�?���{��Q��E�C�K�?���p��_\)��(�C��q                                    BxrIϘ  T          @أ�?��R��=q����(�C�aH?��R���R�l����\C���                                    BxrI�>  T          @�G�?�ff���H<�>uC���?�ff����N�R��=qC��                                    BxrI��  �          @���@G��������3�
C���@G���p��^{���C�33                                    BxrI��  �          @��@�R���>L��?�C���@�R���\�<����ffC�                                      BxrJ
0  
�          @ڏ\@���θR��(��g�C�,�@�������e���
=C��3                                    BxrJ�  �          @�=q@33��
=�#�
��(�C�Ǯ@33�����p����C��\                                    BxrJ'|  �          @��@(����ͿG���33C�xR@(�����w
=�
  C���                                    BxrJ6"  �          @��@�\��(��(���z�C�޸@�\��\)�l���C�Ф                                    BxrJD�  �          @ٙ�@���ff����|(�C���@�����g
=� �\C���                                    BxrJSn  �          @ٙ�@33���
����=qC���@33��G��e��G�C�                                    BxrJb  �          @���@z��ʏ\����  C��@z���p������p�C�g�                                    BxrJp�  �          @�  ?�����ÿ�ff�Tz�C�XR?����p���
=�&p�C���                                    BxrJ`  �          @أ�@  ��G��xQ��  C���@  ��
=�~�R��C�,�                                    BxrJ�  �          @أ�@����ÿ����C��
@������hQ��{C���                                    BxrJ��  �          @׮@�R�ə��W
=��C�@�R�����xQ����C��
                                    BxrJ�R  �          @�\)@!���ff�&ff��G�C�
@!���G��j�H�
=C�O\                                    BxrJ��  �          @�ff@���
=�Y�����C�H�@����R�vff�Q�C���                                    BxrJȞ  &          @�p�@�
��  ��\)�=qC�"�@�
���H��33��\C���                                    BxrJ�D  �          @�p�@������R�+\)C���@���
=��p���C�aH                                    BxrJ��  �          @�@33��{��33�AC�+�@33������=q�!ffC��f                                    BxrJ��  �          @�{?�
=���ÿ��R�+�C��=?�
=������\)�{C���                                    BxrK6  �          @�{?�{��G����
�0z�C�9�?�{��G�������
C��
                                    BxrK�  �          @�ff?�p���  �����>{C��?�p���ff���H�!ffC�l�                                    BxrK �  �          @�@G���{��  �O�
C��@G����H��p��%z�C�޸                                    BxrK/(  �          @�p�?��������{�`(�C���?�����Q���  �*  C���                                    BxrK=�  �          @���?�(���z�Ǯ�Yp�C���?�(�������ff�(=qC���                                    BxrKLt  �          @�z�?����(�������p�C�?�����������9(�C��R                                    BxrK[  �          @�(�?�  �����p���ffC��3?�  ��33��
=�B�RC��                                    BxrKi�  �          @�z�?�33��(���Q�����C���?�33��������7C�~�                                    BxrKxf  �          @�(�?�{��\)���
�1C�J=?�{��\)��Q���
C���                                    BxrK�  �          @�(�@��������HC�O\@���  ���
��RC���                                    BxrK��  �          @�(�@����33��  �
�RC���@����  �|���p�C�c�                                    BxrK�X  �          @��
@��\�u��C��H@���  �z=q��C���                                    BxrK��  �          @Ӆ@.{��p���33�   C�T{@.{�����\)���C��                                    BxrK��  �          @�z�@?\)�������#
=C��@?\)�����}p���\C�)                                    BxrK�J  �          @��@K������{��C�~�@K���(��w��G�C���                                    BxrK��  �          @��@:=q��������
C��@:=q�����y���\)C�Ff                                    BxrK�  �          @��@G����ÿ���=qC�+�@G���p��xQ���C��R                                    BxrK�<  �          @���@J=q��Q�p����\C�aH@J=q��\)�p  �	p�C���                                    BxrL
�  �          @�z�@<�����
�}p��	C�W
@<����G��vff���C��                                     BxrL�  �          @���@;����
���
�\)C�>�@;������x���=qC�t{                                    BxrL(.  �          @��
@z��Å��G����C�h�@z���\)�\)�\)C��                                    BxrL6�  �          @�(�@ ����\)��ff��\C��@ ��������H�ffC�Q�                                    BxrLEz  T          @�p�@�H��(��W
=��C��f@�H���H�w
=��HC�J=                                    BxrLT   �          @�p�@.{���ÿW
=��\)C��@.{��  �s�
�G�C��                                     BxrLb�  �          @�p�@����ͿW
=��  C���@���33�w��{C�O\                                    BxrLql  �          @�@�H��p��W
=��C��
@�H���
�xQ��  C�9�                                    BxrL�  �          @�@*=q�����n{��
=C���@*=q��
=�y�����C���                                    BxrL��  �          @��@'
=�����z�H��HC��@'
=��{�|(��=qC��                                    BxrL�^  �          @��@$z���=q�fff����C�y�@$z���  �x���33C�8R                                    BxrL�  �          @�p�@{���������C��)@{��
=�����  C��3                                    BxrL��  �          @��@���Å��G���C���@����
=��Q��
=C���                                    BxrL�P  �          @�p�@�\�����
�{C�,�@�\��������{C��
                                    BxrL��  �          @�@G��ȣ׿�\)��C���@G�������{�(�C�o\                                    BxrL�  �          @ָR@���G��fff���RC�T{@�����Q���C���                                    BxrL�B  �          @�\)@���=q�J=q��  C���@���  �{��(�C�                                    BxrM�  �          @ָR@��ə��L����(�C�!H@�����|(��\)C�Q�                                    BxrM�  �          @�{?�����p��z�H�33C��?�������������C��                                    BxrM!4  �          @�?�33�˅�5���
C�S3?�33���\�x�����C�C�                                    BxrM/�  �          @���?�z����ÿL����{C�xR?�z����R�{���
C��=                                    BxrM>�  �          @��@���p��O\)�߮C�|)@�����x�����C��R                                    BxrMM&  �          @�z�@�
���0����\)C�:�@�
���r�\�ffC���                                    BxrM[�  �          @�z�@(����Ϳ����G�C��@(���
=�j=q��
C��                                    BxrMjr  �          @�z�@�H��z�:�H��=qC��@�H���
�s�
��C�8R                                    BxrMy  �          @��@����z�fff����C���@�������|����HC�Z�                                    BxrM��  �          @���@���z�Y�����
C���@�����z=q��C�#�                                    BxrM�d  T          @�@ ����p��Ǯ�VffC�@ ����=q�b�\� 
=C�'�                                    BxrM�
  �          @�@ ����{��\)���C��@ ����  �Q����HC���                                    BxrM��  �          @�p�@(����þ�\)��C���@(���
=�`  ��C�^�                                    BxrM�V  �          @�p�@���=q���{�C��@�����k��(�C��)                                    BxrM��  �          @�@����ÿ����\C���@����\�n{�C��R                                    BxrMߢ  T          @��?�33��=q�Y�����C�` ?�33��ff��Q��ffC��H                                    BxrM�H            @�p�@���\)�z�����C��@������o\)��
C�33                                    BxrM��  �          @��?����33�+���=qC�K�?����=q�x�����C�<)                                    BxrN�  �          @���?�����=q�fff����C�#�?�������=q��HC�Ff                                    BxrN:  �          @��@����\)�z�H�
=C�z�@���������\�Q�C��                                    BxrN(�  �          @�?�ff����=�?��
C��H?�ff��(��S�
��=qC��                                     BxrN7�  �          @��?�
=��Q�?z�@���C�%?�
=����:=q��p�C��{                                    BxrNF,  �          @�p�?�  ��<��
>L��C��q?�  ����U��RC��
                                    BxrNT�  �          @�p�@���녿z���  C��@����\�s33�
=C��                                    BxrNcx  �          @�p�?���z��G��xQ�C�Z�?���(��\(���=qC��                                    BxrNr  �          @�@z���33�#�
��C�f@z������U���C�z�                                    BxrN��  �          @�p�?����z�\)��Q�C�7
?�����
�^{����C��=                                    BxrN�j  �          @��?�z���(���Q�:�HC�S3?�z���z��Z�H���HC���                                    BxrN�  �          @�?���p����
�0��C�� ?����\(����
C�9�                                    BxrN��  �          @��?�ff��
=    �#�
C���?�ff��Q��Y����C�ٚ                                    BxrN�\  T          @��?�Q���{<�>�\)C�XR?�Q���  �Vff���
C���                                    BxrN�  �          @���?�=q��z�W
=��C���?�=q���\�a�� p�C�u�                                    BxrNب  �          @�z�@�R�ƸR<#�
=�C��@�R��G��P  ��(�C�p�                                    BxrN�N  �          @�(�@p���
=>��R@.�RC���@p����R�@  ��=qC�H                                    BxrN��  T          @��
@\)�ƸR>��R@.{C��@\)��ff�@  ���C�.                                    BxrO�  T          @�z�?�Q����H=#�
>�{C�}q?�Q�����S33����C��
                                    BxrO@  T          @�z�@
=q��Q�=�?��C���@
=q��z��L(����C��=                                    BxrO!�  �          @�(�?�ff��33>\)?��HC���?�ff��
=�N{���C��                                    BxrO0�  �          @�33?�
=��p�?�R@�C�1�?�
=�����6ff�Ώ\C��                                     BxrO?2  �          @�33?�\)��
=>��@a�C��?�\)���R�Dz���=qC���                                    BxrOM�  �          @Ӆ?�����p�?L��@�ffC��?������
�-p���p�C�s3                                    BxrO\~  �          @��
?�G���ff?uA{C�p�?�G���\)�%����\C��f                                    BxrOk$  �          @�33?�ff���?^�R@�\C���?�ff����)������C�4{                                    BxrOy�  �          @��
?333����?J=q@�33C�{?333���R�1��ƣ�C�^�                                    BxrO�p  �          @Ӆ?0������?.{@���C�  ?0������8Q���z�C�P�                                    BxrO�  �          @�33?�G���(�?0��@�Q�C���?�G������2�\��  C�L�                                    BxrO��  �          @Ӆ?��R���>��H@���C��\?��R��ff�>{��  C�XR                                    BxrO�b  �          @��H?�G��ʏ\?
=q@�\)C���?�G�����8����(�C�}q                                    BxrO�  �          @ҏ\@����p�>.{?�(�C��@����=q�G����HC��H                                    BxrOѮ  �          @�=q?�ff��  �&ff���C�
=?�ff���R�w
=���C���                                    BxrO�T  �          @ҏ\?�33��Q��ff�|��C�n?�33��=q�l���	\)C�B�                                    BxrO��  �          @ҏ\?���ʏ\�Y������C�ٚ?����p���=q�(�C��{                                    BxrO��  �          @Ӆ@7����?.{@��C���@7����
�#33��  C��                                    BxrPF  �          @��
@L(����R?�G�A  C���@L(���(��
�H��\)C�Y�                                    BxrP�  �          @�33@8�����H?uA�RC�*=@8�����R����  C��                                    BxrP)�  �          @��H@E���p�?��A4��C�=q@E��������p�C��H                                    BxrP88  �          @ҏ\?޸R��=q������C��{?޸R����o\)�  C�j=                                    BxrPF�  �          @Ӆ@z���  ��G��s33C�,�@z�����l(����C�!H                                    BxrPU�  T          @Ӆ@G���Q�����C��3@G������q���\C��
                                    BxrPd*  �          @��H?�Q����H�z����\C�}q?�Q�����w
=���C�C�                                    BxrPr�  �          @��H?�����녿�\��p�C��?�����=q�q��=qC�޸                                    BxrP�v  �          @�33@G��ȣ׾����:=qC��=@G���(��g��
=C���                                    BxrP�  
�          @��H@z���  ��Q��EC�.@z���33�g��ffC�\                                    BxrP��  �          @��H?�����ÿ����C�\)?�������u��33C�P�                                    BxrP�h  �          @�33?�\)�ə�������C�Ff?�\)��G��tz����C�0�                                    BxrP�  �          @��H?�ff��녿����C��
?�ff�����u��
C��
                                    BxrPʴ  �          @�=q?�����G��z�����C��?��������u��HC��R                                    BxrP�Z  �          @�=q?�G��ə��������C��\?�G���Q��w��G�C��\                                    BxrP�   �          @љ�?޸R��G�����ffC�� ?޸R����p���(�C��                                     BxrP��  �          @љ�?�{��=q��\��\)C�*=?�{��=q�s33�{C��3                                    BxrQL  �          @��?�����녿Q���RC��q?������������C���                                    BxrQ�  �          @љ�?��ȣ׿c�
����C�y�?����\���H�C��                                    BxrQ"�  �          @��?�{��=q�&ff���C�&f?�{��  �z�H��HC��                                    BxrQ1>  �          @��?�G��ȣ׿:�H��(�C���?�G�����}p����C�޸                                    BxrQ?�  �          @ҏ\?�����녾�
=�j�HC��?�������n{�	�C��                                    BxrQN�  �          @��H?�z���33�����C�P�?�z����\�u��
C��                                    BxrQ]0  �          @��H?�=q���;�=q���C��?�=q��Q��i���p�C�U�                                    BxrQk�  �          @ҏ\?����ȣ׾�ff�|��C���?��������n�R�
33C���                                    BxrQz|  �          @��H?����(��L�;��C�33?�����
�\�����RC�|)                                    BxrQ�"  �          @��H?�
=�˅����\)C�c�?�
=����[���33C���                                    BxrQ��  �          @��H?���녾��
�4z�C��?�����i�����C��                                    BxrQ�n  �          @��@ ���Ǯ=�G�?�  C��@ �����\�O\)����C�T{                                    BxrQ�  �          @��@���\)��\)�&ffC�  @���
=�Y�����
C��R                                    BxrQú  �          @љ�@�����#�
��p�C��)@����U���G�C�ff                                    BxrQ�`  �          @��@$z���\)?5@�{C���@$z����'
=����C��                                    BxrQ�  �          @ҏ\@#33����=���?h��C�k�@#33��p��I����p�C�                                      BxrQ�  �          @���@5����?��
AV�\C�
@5��ff��ff�|(�C�=q                                    BxrQ�R  �          @�(�@J=q���
@$z�A��HC��R@J=q��{������C���                                    BxrR�  
�          @��
@(���p�?Q�@�C��@(������'����HC��R                                    BxrR�  �          @�(�@
=��Q�L�Ϳ�p�C�S3@
=���aG�� �
C�R                                    BxrR*D  �          @��
?�z����ÿG��أ�C�|)?�z���(��������C���                                    BxrR8�  �          @��H?���ȣ׿������C�W
?����{����#C���                                    BxrRG�  �          @��H?�33��{��p��tz�C�` ?�33������G��:�C��                                    BxrRV6  �          @�33?�����33�
�H��{C��\?����\)����KG�C�`                                     BxrRd�  �          @��H?�=q��p��������C�3?�=q��ff�����@(�C�Ǯ                                    BxrRs�  �          @ҏ\?�������{��\)C�^�?����ff���
�?{C�+�                                    BxrR�(  �          @Ӆ?��
��p����H��Q�C��
?��
�������R�C=qC���                                    BxrR��  �          @Ӆ?ٙ���
=�����K�C��?ٙ����R����.C�l�                                    BxrR�t  �          @�(�?��H�Ǯ���F�\C���?��H��  �����-�\C�h�                                    BxrR�  �          @���?����  ���\�0(�C�l�?�����H��p��&�HC�33                                    BxrR��  �          @���?�
=�Ǯ��(��)��C���?�
=�����(��%  C�`                                     BxrR�f  
�          @�z�?���ȣ׿�����C�W
?������=q�"ffC��\                                    BxrR�  �          @���?��H��=q��{�{C��?��H������\�"�HC��=                                    BxrR�  �          @��?�  ��녿��H�'33C�?�  ������%��C�Ff                                    BxrR�X  �          @�p�?��H��=q���R�+
=C��
?��H��p���{�'  C��                                    BxrS�  �          @��?�{���ÿ�(��(z�C�>�?�{��z������%p�C���                                    BxrS�  T          @��?�{���ÿ����&ffC�<)?�{������z��$�C��H                                    BxrS#J  �          @�p�?�ff���H��ff��C��=?�ff������G�� =qC�N                                    BxrS1�  �          @��?�Q����H����=qC�y�?�Q���\)���
�$�C��
                                    BxrS@�  �          @���?�(���33�z�H��C��R?�(���=q����ffC��3                                    BxrSO<  �          @ҏ\?�z��ə��^�R��z�C�` ?�z����H���
�p�C�t{                                    BxrS]�  �          @ҏ\?��
�ʏ\�p���33C�Ǯ?��
���\��{�z�C���                                    BxrSl�  �          @��?����ff�z�H�	�C�q�?����ff�����RC��=                                    BxrS{.  T          @љ�@G���(�������RC�#�@G���=q��
=� �C��\                                    BxrS��  �          @��H?����(���p��P(�C��f?�����
�����.��C���                                    BxrS�z  �          @�33?����p������<��C���?����
=��ff�*
=C�y�                                    BxrS�   �          @Ӆ?�  �Ǯ��  �/33C�Ф?�  ���\����'�HC�l�                                    BxrS��  T          @�33?�ff��녿����{C��?�ff���R��33�%
=C��                                    BxrS�l  �          @��?�\)��zΰ�
�1�C��?�\)��ff��Q��*�
C��                                    BxrS�  �          @���?�\)�������9�C��{?�\)�����\�.33C���                                    BxrS�  �          @���?333�θR���E�C��?333����p��2�\C�9�                                    BxrS�^  �          @���?��Ϯ��=q�8��C��?���  ����/C�n                                    BxrS�  �          @��?   ���ÿ�z��!G�C�5�?   ��(���  �*{C��{                                    BxrT�  �          @���>��љ�������C��>����R�����%��C��                                     BxrTP  �          @��
?��У׿u���C��H?���\)��=q�"�
C�E                                    BxrT*�  T          @��H>����G��0����  C�3>����z���33��
C���                                    BxrT9�  �          @�33?��љ��������C�� ?���\)�\)��HC�"�                                    BxrTHB  �          @У�?J=q���!G����C��H?J=q���\�\)�\)C�k�                                    BxrTV�  �          @�  ?���  �   ��C�~�?�����r�\���C�B�                                    BxrTe�  �          @�
=?�{�ʏ\�#�
���RC��R?�{����|�����C�E                                    BxrTt4  �          @�
=?s33���
�
=��
=C�C�?s33�����z�H�p�C�Z�                                    BxrT��  �          @θR?������H���R�333C��=?�����p��k��G�C��q                                    BxrT��  �          @�ff?���ə��#�
��p�C��q?����G��[�� ��C��                                    BxrT�&  �          @�p�?�  �Ǯ��Q�G�C���?�  ���R�[��\)C��                                    BxrT��  �          @���@	����Q�#�
��{C���@	�������QG���Q�C��                                    BxrT�r  �          @���@'
=��=q����
=C�
@'
=��Q��Vff���C�`                                     BxrT�  �          @�@P����Q쾀  �\)C�P�@P����Q��K���C���                                    BxrTھ  �          @���@.{��G�<�>�  C���@.{��z��E��p�C���                                    BxrT�d  T          @���@����;����e�C�5�@���Q��`���G�C���                                    BxrT�
  �          @��@\)���;�\)�   C�l�@\)��=q�Z=q� �C���                                    BxrU�  �          @�p�@ff��
=�\)���HC��H@ff���R�U����
C���                                    BxrUV  �          @��@(��������
=C�<)@(����H�Y��� z�C�ff                                    BxrU#�  �          @���@�������H���
C��@������hQ��
�\C�C�                                    BxrU2�  �          @���@@����(�>�{@EC��@@�����/\)���C��\                                    BxrUAH  �          @�(�@AG���33=�?���C�,�@AG������:=q�مC�!H                                    BxrUO�  �          @�z�@Vff��=�G�?s33C��{@Vff��z��5����HC���                                    BxrU^�  �          @�z�@1�����u�
=qC��=@1����R�R�\��ffC�E                                    BxrUm:  �          @���@&ff��=q�����\)C��@&ff�����a��
=C��)                                    BxrU{�  �          @��@.{��G��\�X��C��{@.{���[����C�
                                    BxrU��  �          @�@,(����������C�t{@,(���(��a��p�C�R                                    BxrU�,  �          @�@2�\��Q��\����C��\@2�\���H�a��33C��                                    BxrU��  T          @���@4z���
=��\��=qC�&f@4z���G��`����C��                                    BxrU�x  �          @�z�@.{���׾�\)�"�\C���@.{���R�U���\C���                                    BxrU�  �          @�p�@C�
��녿k��z�C�g�@C�
��{�qG��  C�\                                    BxrU��  �          @�(�@)�����þ��H���C�P�@)������aG���C���                                    BxrU�j  �          @�(�@5���ff������RC�/\@5�����]p��ffC���                                    BxrU�  �          @���@<(���������C���@<(������[����C�}q                                    BxrU��  �          @���@�����
�+����C�R@����33�n{���C��                                    BxrV\  �          @���?�33�����E���p�C��{?�33��{�y�����C��                                    BxrV  �          @���@H����=q����=qC���@H����(��E�癚C��                                    BxrV+�  �          @�p�@hQ���G�>W
=?���C�:�@hQ���=q�*�H��z�C�8R                                    BxrV:N  �          @���@|(���Q�?��@���C��@|(�������R��G�C���                                    BxrVH�  �          @�@w����>�ff@�  C��{@w����������HC�4{                                    BxrVW�  T          @���@s�
��(�>�ff@�  C�K�@s�
��=q�Q���ffC���                                    BxrVf@  �          @���@xQ�����?=p�@�(�C��=@xQ���������C��                                    BxrVt�  �          @˅@ ����p��
=����C���@ ����
=�b�\�
�\C��                                    BxrV��  �          @�z�?������H��ff�<z�C�e?�����{���
�,�C��f                                    BxrV�2  �          @��?�  ��G���p��z�HC�޸?�  ��p���ff�<�C�b�                                    BxrV��  �          @��?�G����ÿ�G���C���?�G���z���
=�>
=C�q�                                    BxrV�~  �          @��?����=q��\��Q�C��R?����p���  �?z�C���                                    BxrV�$  �          @���?Q����R�
�H��ffC��?Q��w������OG�C��R                                    BxrV��  �          @��?+�����
�H��ffC�7
?+��x�������O�
C��                                    BxrV�p  �          @��?!G���\)�	����G�C��)?!G��x�������O\)C��\                                    BxrV�  �          @��?������R��Q��PQ�C���?�����Q�����.p�C��R                                    BxrV��  �          @�{@/\)����z�H���C�Ǯ@/\)��=q�z=q��C�9�                                    BxrWb  
\          @�@'
=��=q�J=q��\C��@'
=��\)�r�\�z�C�{                                    BxrW  
�          @��@"�\���H�5���HC���@"�\�����n�R�=qC���                                    BxrW$�  
�          @�z�@"�\��녿0����C�Ф@"�\��G��l(���C��H                                    BxrW3T  �          @�z�@P���������hQ�C�~�@P�����
�P  ��Q�C�]q                                    BxrWA�  "          @��@^{�����8Q��\)C��q@^{���H�HQ���z�C���                                    BxrWP�  	�          @�G�@r�\����>��?�{C��@r�\����-p���Q�C���                                    BxrW_F  
Z          @�G�@�����H?�@��C�K�@�����\��\���C�˅                                    BxrWm�  
�          @ҏ\@�p����?��@�
=C�@�p����H�����=qC�                                      BxrW|�  
Z          @У�@�������?Y��@�G�C��@�����������C��                                    BxrW�8  T          @У�@�G���z�?�G�AV�\C��R@�G������p�����C�
=                                    BxrW��  
�          @Ϯ@������?�A&=qC���@�������ff�8��C��                                    BxrW��  
�          @�p�@����ff?���A{C��H@�����H��(��Tz�C��                                    BxrW�*  �          @�@�����\)?0��@�(�C��@������
������G�C�E                                    BxrW��  	�          @���@��\��>aG�?��HC�˅@��\���������{C�˅                                    BxrW�v  �          @���@���=q>��@�  C�w
@����\����
=C�H                                    BxrW�  
�          @��@�p���33>�Q�@P��C�S3@�p���G������C�\                                    BxrW��  
�          @�{@U��\)>�@��\C��@U��(��"�\��  C�4{                                    BxrX h  	�          @�p�@c33��Q�?^�R@���C���@c33��p�����33C��                                    BxrX  T          @˅@�����\?Q�@�C��@�����׿����=qC��3                                    BxrX�  T          @��
@�����Q�?Q�@�33C��@������R�������C�z�                                    BxrX,Z  �          @˅@����?G�@ᙚC�R@�����
��=q��(�C�                                    